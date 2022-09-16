import os
import numpy as np
import pickle as pkl
import torch
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.tools.model_loader import load_model
import yaml
from os.path import join as ospj
from pytorch3d.transforms import rotation_conversions as p3d_rt
from tqdm import tqdm
from mcms.dsets import h36m, copenet_real, rich
from savitr_pe.datasets import savitr_dataset
import cv2
from mcms.utils.renderer import Renderer
from human_body_prior.body_model.body_model import BodyModel
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

res_file = "/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/copenet_zLatent/0000/stage_01/_seq_start_00005.pkl"
vizdir = "copenet_overlays"
res_dir = "/".join(res_file.split("/")[:-3])
config = yaml.safe_load(open(ospj(res_dir,"0000","config.yml")))
dset = config["dset"]
hparams = yaml.safe_load(open("/".join(config["mo_vae_ckpt_path"].split("/")[:-2])+"/hparams.yaml","r"))
device = torch.device(config["device"])
big_seq_start = config["big_seq_start"]


# Dataloader
if dset.lower() == "h36m":
    ds = h36m.h36m(hparams,used_cams=config["cams_used"])
    fps_scl = 2
    viz_dwnsample = 1
    # renderer
    im_res = [[1000,1000],[1000,1000],[1000,1000],[1000,1000]]
elif dset.lower() == "copenet_real":
    hparams["data_datapath"] = config["data_path"]
    ds = copenet_real.copenet_real(hparams,range(0,7000))
    fps_scl = 1
    viz_dwnsample = 1
    im_res=[[1920,1080],[1920,1080]]
elif dset.lower() == "savitr":
    ds = savitr_dataset.savitr_dataset(config["data_path"],seq_len=25)
    im_lists = ds.__getitem__(10,seq_len=1)["full_im_paths"]
    im_res = [[cv2.imread(i[0]).shape[1],cv2.imread(i[0]).shape[0]] for i in im_lists]
    fps_scl = 1
    viz_dwnsample = 1
elif dset.lower() == "rich":
    # "/ps/project/datasets/AirCap_ICCV19/RICH_IPMAN/test/2021-06-15_Multi_IOI_ID_00186_Yoga1"
    ds = rich.rich(config["data_path"],used_cams=config["cams_used"])
    im_res = ds.im_res[:,::-1]
    fps_scl = 1
    viz_dwnsample = 5
renderer = [Renderer(img_res=[np.ceil(res[0]),np.ceil(res[1])]) for res in im_res]

# SMPL
smpl = BodyModel(bm_fname=hparams["model_smpl_neutral_path"]).to(device)

# vposer model
vp_model = load_model(hparams["model_vposer_path"], model_code=VPoser,remove_words_in_model_weights="vp_model.",map_location=device)[0].to(device)
vp_model.eval()

batch = ds.__getitem__(big_seq_start)
j2ds = batch["j2d"].float().to(device)
num_cams = j2ds.shape[0]
cam_intr = torch.from_numpy(batch["cam_intr"]).float().to(device)



def viz_results(stage_dict):

    j2ds = stage_dict["j2ds"]
    full_images_paths = stage_dict["full_im_paths"]
    curr_seq_len = j2ds.shape[1]

    smpl_trans = stage_dict["smpl_trans"]
    smpl_orient = stage_dict["smpl_orient"]
    smpl_art_motion_vp_latent = stage_dict["smpl_art_motion_vp_latent"]
    smpl_shape = stage_dict["smpl_shape"]

    
    cam_orient_expanded = torch.cat([x.repeat(1,curr_seq_len,1,1) if x.shape[1]==1 else x for x in stage_dict["cam_orient"]],dim=0)
    cam_position_expanded = torch.cat([x.repeat(1,curr_seq_len,1) if x.shape[1]==1 else x for x in stage_dict["cam_position"]],dim=0)
    cam_ext_temp = torch.cat([cam_orient_expanded,cam_position_expanded.unsqueeze(-1)],dim=3)
    cam_ext = torch.cat([cam_ext_temp,torch.tensor([0,0,0,1]).type_as(cam_ext_temp).repeat(num_cams,curr_seq_len,1,1)],dim=2)
    cam_poses = torch.inverse(cam_ext)
    
    smpl_art_motion_interm = torch.cat([vp_model.decode(smpl_art_motion_vp_latent[i:i+1])["pose_body"] for i in range(smpl_art_motion_vp_latent.shape[0])])
    smpl_motion = torch.cat([smpl_trans.unsqueeze(1),p3d_rt.matrix_to_axis_angle(smpl_orient).unsqueeze(1),
                                    smpl_art_motion_interm],dim=1).reshape(curr_seq_len,69)

    # SMPL fwd pass
    smpl_motion_list = [smpl_motion[100*i:100*(i+1)] for i in range(smpl_motion.shape[0]//100)]
    if smpl_motion.shape[0]%100 > 0:
        smpl_motion_list.append(smpl_motion[100*(smpl_motion.shape[0]//100):])
    smpl_out_v = [smpl.forward(root_orient = x[:,3:6],
                                pose_body = x[:,6:],
                                trans = x[:,:3],
                                betas = smpl_shape.unsqueeze(1).expand(-1,x.shape[0],-1).reshape(-1,smpl_shape.shape[-1])).v
                                for x in smpl_motion_list]
    smpl_out_v = torch.cat(smpl_out_v,dim=0)

    t_range = range(0,curr_seq_len,10)
    orig_ims_list = []
    rend_ims_list = []
    cat_ims_list = []
    for cam in tqdm(range(num_cams)):
        orig_ims = []
        rend_ims = []
        cat_ims = []
        for s in tqdm(t_range):
            im = cv2.imread(full_images_paths[cam][s])[:im_res[cam][1],:im_res[cam][0],::-1]/255.
            temp_intr = cam_intr[cam].clone().detach().cpu().numpy()
            temp_v = smpl_out_v.view(curr_seq_len,6890,3)[s].clone().detach().cpu().numpy()
            bb_min = j2ds[cam,s,j2ds[cam,s,:,2]!=0,:2].min(dim=0).values.int() - 50
            bb_min[bb_min<0] = 0
            bb_max = j2ds[cam,s,j2ds[cam,s,:,2]!=0,:2].max(dim=0).values.int() + 50
            
            rend_im = renderer[cam](temp_v,
                                cam_ext[cam,s,:3,3].clone().detach().cpu().numpy(),
                                cam_ext[cam,s,:3,:3].unsqueeze(0).clone().detach().cpu().numpy(),
                                im,intr=temp_intr,
                                faces=smpl.f.clone().detach().cpu().numpy(),color=(0.3,0.8,0.8,0.5))
            rend_im = rend_im[bb_min[1] : bb_max[1], bb_min[0] : bb_max[0]]
            im = im[bb_min[1] : bb_max[1], bb_min[0] : bb_max[0]]
            scale = 300.0/rend_im.shape[0]
            orig_ims.append(cv2.copyMakeBorder(cv2.resize(im,(int(im.shape[1]*scale),300)),1,2,1,2,cv2.BORDER_CONSTANT,value=[0,0,0]))
            rend_ims.append(cv2.copyMakeBorder(cv2.resize(rend_im,(int(rend_im.shape[1]*scale),300)),1,2,1,2,cv2.BORDER_CONSTANT,value=[0,0,0]))
            cat_ims.append(np.concatenate([orig_ims[-1],rend_ims[-1]],axis=0))
        
        orig_ims_list.append(orig_ims)
        rend_ims_list.append(rend_ims)
        cat_ims_list.append(cat_ims)
    os.makedirs("/is/ps3/nsaini/projects/mcms/mcms_logs/paper_mat/{}".format(vizdir),exist_ok=True)
    import ipdb;ipdb.set_trace()
    for i in range(len(cat_ims_list[0])):
        cv2.imwrite("/is/ps3/nsaini/projects/mcms/mcms_logs/paper_mat/{}/{:04d}.png".format(vizdir,t_range[i]),
        cv2.copyMakeBorder(np.concatenate([x[i] for x in cat_ims_list],axis=1),2,2,2,2,cv2.BORDER_CONSTANT,value=[255,255,255])[:,:,::-1]*255)
    
                
# max_width = np.max([x.shape[1] for x in orig_ims])
# orig_ims = [torch.from_numpy(cv2.copyMakeBorder(x,0,0,(max_width-x.shape[1])//2,max_width-x.shape[1]-(max_width-x.shape[1])//2,cv2.BORDER_CONSTANT,value=[255,255,255])).permute(2,0,1) for x in orig_ims]
# rend_ims = [torch.from_numpy(cv2.copyMakeBorder(x,0,0,(max_width-x.shape[1])//2,max_width-x.shape[1]-(max_width-x.shape[1])//2,cv2.BORDER_CONSTANT,value=[255,255,255])).permute(2,0,1) for x in rend_ims]
# orig_grid = make_grid(orig_ims).permute(1,2,0).detach().cpu().numpy()
# rend_grid = make_grid(rend_ims).permute(1,2,0).detach().cpu().numpy()

if __name__ == "__main__":

    res_stage = pkl.load(open(res_file,"rb"))
    viz_results(res_stage)