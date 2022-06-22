import os
from sqlite3 import PARSE_COLNAMES
from warnings import catch_warnings
os.environ["PYOPENGL_PLATFORM"] = 'egl'
import yaml
import torch
from pytorch3d.transforms import rotation_conversions as p3d_rt
import imageio
from torchvision.utils import make_grid
import glob
import sys

from mcms.dsets import h36m, copenet_real, rich
from savitr_pe.datasets import savitr_dataset
from torch.utils.data import DataLoader
from nmg.models import nmg
import numpy as np
from tqdm import tqdm, trange
from mcms.utils.utils import nmg2smpl, smpl2nmg
from smplx.body_models import create
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.tools.model_loader import load_model
from mcms.utils.geometry import perspective_projection
from mcms.utils.renderer import Renderer
import cv2
from mcms.utils import geometry
from mcms.utils.utils import to_homogeneous, gmcclure
import sys
from torch.utils.tensorboard import SummaryWriter
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
import pickle as pkl
from torch.optim.lr_scheduler import ReduceLROnPlateau


trial_name = sys.argv[1]
seq_no = sys.argv[2]

resume = False
# check if the trial exists
if os.path.exists(os.path.join("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings",trial_name)):
    config = yaml.safe_load(open(os.path.join("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings",
                trial_name,
                "{:04d}".format(int(seq_no)),
                "config.yml")))
    stage_dirs = sorted(glob.glob(os.path.join("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings",
                trial_name,
                "{:04d}".format(int(seq_no)),
                "stage_*")))
    if len(stage_dirs) > 0:
        resume = True
else:
    print("\n Trial doesn't exists, starting new trial")
    config = yaml.safe_load(open("/is/ps3/nsaini/projects/mcms/src/mcms/fitting_scripts/fit_config.yml"))

config["trial_name"] = trial_name
config["seq_no"] = seq_no

if config["dset"].lower() != "h36m":
    seq_no = 0

# config parameters
batch_size = config['batch_size']
nmg_seq_len = config["nmg_seq_len"]
init_seq_len = config["init_seq_len"]
loss_2d_weight = config["loss_2d_weight"]
loss_z_weight = config["loss_z_weight"]
loss_cams_orient_weight = config["loss_cams_orient_weight"]
loss_cams_position_weight = config["loss_cams_position_weight"]
loss_betas_weight = config["loss_betas_weight"]
n_optim_iters = config["n_optim_iters"]
loss_human_gp_weight = config["loss_human_gp_weight"]
loss_cam_gp_weight = config["loss_cam_gp_weight"]
loss_vp_weight = config["loss_vp_weight"]
loss_smpl_in_front_weight = config["loss_smpl_in_front_weight"]
loss_j3d_smooth_weight = config["loss_j3d_smooth_weight"]
lr = config["lr"]
dset = config["dset"]
seq_no = config["seq_no"]
big_seq_start = config["big_seq_start"]
big_seq_end = config["big_seq_end"]
overlap = config["overlap"]
hparams = yaml.safe_load(open("/".join(config["mo_vae_ckpt_path"].split("/")[:-2])+"/hparams.yaml","r"))
device = torch.device(config["device"])

# SMPL
smpl = BodyModel(bm_fname=hparams["model_smpl_neutral_path"]).to(device)

# Motion VAE
nmg_hparams = yaml.safe_load(open("/".join(hparams["train_motion_vae_ckpt_path"].split("/")[:-2])+"/hparams.yaml"))
mvae_model = nmg.nmg.load_from_checkpoint(hparams["train_motion_vae_ckpt_path"],map_location=device).to(device)
mean_std = np.load(hparams["model_mvae_mean_std_path"])
mvae_mean = torch.from_numpy(mean_std["mean"]).float().to(device)
mvae_std = torch.from_numpy(mean_std["std"]).float().to(device)

# vposer model
vp_model = load_model(hparams["model_vposer_path"], model_code=VPoser,remove_words_in_model_weights="vp_model.",map_location=device)[0].to(device)
vp_model.eval()

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
    im_lists = ds.__getitem__(1450,seq_len=1)["full_im_paths"]
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

# make dir for seq_no
if dset.lower() == "h36m":
    os.makedirs("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}".format(config["trial_name"],seq_no),exist_ok=True)
else:
    seq_no = 0
    os.makedirs("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}".format(config["trial_name"],seq_no),exist_ok=True)

if dset.lower() == "h36m":
    batch = ds.__getitem__(seq_no,big_seq_start)
else:
    batch = ds.__getitem__(big_seq_start)
j2ds = batch["j2d"].float().to(device)
num_cams = j2ds.shape[0]
cam_intr = torch.from_numpy(batch["cam_intr"]).float().to(device)


def load_batch_wth_pare_init():
    # dump yaml file
    yaml.safe_dump(config,open("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/config.yml".format(config["trial_name"],seq_no),"w"))

    # init_cam_position = []
    # init_cam_orient = []
    # init_smpl_trans_rf = []
    # init_smpl_orient_rf = []
    # init_smpl_trans_ff = []
    # init_smpl_orient_ff = []
    # init_smpl_art_motion_vp_latent = []
    # init_smpl_shape = []
    pare_init_state = []

    init_stage_j2d = []
    init_stage_im_paths = []
    with trange(big_seq_start,big_seq_end,fps_scl*(init_seq_len-overlap)) as seq_t:
        for seq_start in seq_t:

            # tensorboard summarywriter
            # os.makedirs("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/{:04d}".format(config["trial_name"],seq_no,seq_start),exist_ok=True)
            # writer = SummaryWriter(log_dir="/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/{:04d}".format(config["trial_name"],seq_no,seq_start), 
            #                 filename_suffix="{:04d}".format(seq_start))

            if dset.lower() == "h36m":
                # get batch
                batch = ds.__getitem__(seq_no,seq_start)
            else:
                batch = ds.__getitem__(seq_start)
            j2ds = batch["j2d"].float().to(device)
            init_stage_j2d.append(j2ds.clone().detach())
            init_stage_im_paths.append(batch["full_im_paths"])
            
            smpl_shape = torch.zeros(10).unsqueeze(0).to(device)

            mvae_model.eval()
            smpl_motion_init = nmg2smpl((mvae_model.decode(torch.zeros(1,1024).to(device))*mvae_std + mvae_mean).reshape(nmg_seq_len,22,9),smpl)
            smpl_trans = smpl_motion_init[:,:3].clone().detach()
            smpl_orient = p3d_rt.axis_angle_to_matrix(smpl_motion_init[:,3:6])
            smpl_art_motion_vp_latent_init = torch.cat([vp_model.encode(batch["pare_poses"][j, :21].reshape(1, 63).to(device)).mean for j in range(init_seq_len)]).clone().detach()
            # first frame init
            pare_orient = p3d_rt.axis_angle_to_matrix(batch["pare_orient"].to(device))
            pare_cams = batch["pare_cams"].to(device)

            cam_orient = torch.matmul(pare_orient[:,0],torch.inverse(smpl_orient[0:1])).unsqueeze(1)
            cam_position = torch.matmul(pare_orient[:,0],
                                torch.matmul(torch.inverse(smpl_orient[0:1]), smpl_trans[0:1].unsqueeze(-1))) \
                                + pare_cams[:, 0].unsqueeze(-1)
            cam_position = [cam_position[c].squeeze(-1).unsqueeze(0).repeat([1,init_seq_len,1]).clone().detach() if c in config["position_changing_cams"] 
                            else cam_position[c].squeeze(-1).unsqueeze(0).unsqueeze(0).clone().detach() for c in range(num_cams)]
            cam_orient = [cam_orient[c].unsqueeze(0).repeat([1,init_seq_len,1,1]).clone().detach() if c in config["orient_changing_cams"] 
                            else cam_orient[c,0:1].unsqueeze(0).clone().detach() for c in range(num_cams)]
            smpl_orient = smpl_orient.clone().detach()

            pare_init_state.append({"cam_orient":cam_orient, "cam_position": cam_position,
                        "smpl_trans":smpl_trans, 
                        "smpl_orient": smpl_orient, "smpl_shape": smpl_shape, 
                        "smpl_art_motion_vp_latent":smpl_art_motion_vp_latent_init, "j2ds":j2ds.clone().detach(), 
                        "full_im_paths": batch["full_im_paths"]})

    return pare_init_state


def motion2nmg(strt_idx,smpl_trans,smpl_orient,smpl_motion):
    global smpl
    curr_pos, curr_ori = geometry.get_ground_point(smpl_trans[strt_idx],p3d_rt.rotation_6d_to_matrix(smpl_orient[strt_idx]))
    curr_tfm = to_homogeneous(curr_ori,curr_pos)
    curr_root_pose = to_homogeneous(p3d_rt.rotation_6d_to_matrix(smpl_orient[strt_idx:strt_idx+nmg_seq_len]),smpl_trans[strt_idx:strt_idx+25])
    canonical_root_pose = torch.matmul(torch.inverse(curr_tfm),curr_root_pose)
    canonical_smpl_motion = torch.cat([canonical_root_pose[:,:3,3],
                            p3d_rt.matrix_to_axis_angle(canonical_root_pose[:,:3,:3]),
                            smpl_motion[strt_idx:strt_idx+nmg_seq_len,6:]],dim=1)
    nmg_repr = (smpl2nmg(canonical_smpl_motion,smpl).reshape(-1,25,22*9) - mvae_mean)/mvae_std

    return nmg_repr


def optimize(params,device, iters):
    
    global smpl
    global vp_model
    global mvae_model
    global mvae_mean
    global mvae_std
    global cam_intr

    smpl = smpl.to(device)
    vp_model = vp_model.to(device)
    mvae_model = mvae_model.to(device)
    mvae_mean = mvae_mean.to(device)
    mvae_std = mvae_std.to(device)

    cam_orient = [p3d_rt.matrix_to_rotation_6d(x).clone().detach().to(device).requires_grad_(True) for x in params["cam_orient"]]
    cam_position = [x.clone().detach().to(device).requires_grad_(True) for x in params["cam_position"]]
    smpl_trans_ff = params["smpl_trans"][:1].clone().detach().to(device).requires_grad_(True)
    smpl_orient_ff = p3d_rt.matrix_to_rotation_6d(params["smpl_orient"][:1]).clone().detach().to(device).requires_grad_(True)
    smpl_trans_rf = params["smpl_trans"][1:].clone().detach().to(device).requires_grad_(True)
    smpl_orient_rf = p3d_rt.matrix_to_rotation_6d(params["smpl_orient"][1:]).clone().detach().to(device).requires_grad_(True)
    smpl_shape = params["smpl_shape"].clone().detach().to(device).requires_grad_(True)
    smpl_art_motion_vp_latent = params["smpl_art_motion_vp_latent"].clone().detach().to(device).requires_grad_(True)
    j2ds = params["j2ds"].clone().detach().to(device)
    cam_intr = cam_intr.to(device)
    curr_seq_len = j2ds.shape[1]

    optim0 = torch.optim.Adam(cam_orient + cam_position,lr=lr[0])
    optim1 = torch.optim.Adam([{"params": cam_position + cam_orient, "lr": lr[1]*10}, {"params": [smpl_trans_rf,smpl_orient_rf]}],lr=lr[1])
    optim2 = torch.optim.Adam([{"params": cam_position + cam_orient, "lr": lr[2]*10} , {"params": [smpl_art_motion_vp_latent, smpl_trans_rf, smpl_orient_rf, 
                                    smpl_shape]}],lr=lr[2])
    sched0 = ReduceLROnPlateau(optim0, 'min')
    sched1 = ReduceLROnPlateau(optim1, 'min')
    sched2 = ReduceLROnPlateau(optim2, 'min')

    with trange(iters[2]) as t:
        for i in t:
            
            if i < iters[0]:
                stage = 0
                optim = optim0
                sched = sched0
            elif i >= iters[0] and i < iters[1]:
                stage = 1
                optim = optim1
                sched = sched1
            else:
                stage = 2
                optim = optim2
                sched = sched2
            
            ################### FWD pass #####################
            # smpl_art_motion_interm = torch.cat([smpl_art_motion[:,:9],smpl_art_motion_nonOpt[:,:3],
            #                                     smpl_art_motion[:,9:11],smpl_art_motion_nonOpt[:,3:],
            #                                     smpl_art_motion[:,11:]],dim=1)
            
            smpl_art_motion_interm = vp_model.decode(smpl_art_motion_vp_latent)["pose_body"]
            smpl_trans = torch.cat([smpl_trans_ff,smpl_trans_rf])
            smpl_orient = torch.cat([smpl_orient_ff,smpl_orient_rf])
            
            smpl_motion = torch.cat([smpl_trans.unsqueeze(1),p3d_rt.matrix_to_axis_angle(p3d_rt.rotation_6d_to_matrix(smpl_orient)).unsqueeze(1),
                                    smpl_art_motion_interm],dim=1).reshape(curr_seq_len,69)
            # Decode smpl motion using motion vae
            mvae_model.eval()
            nmg_repr_list = Parallel(n_jobs=-1)(delayed(motion2nmg)(strt_idx,smpl_trans,smpl_orient,smpl_motion) 
                            for strt_idx in range(0,curr_seq_len-nmg_seq_len+1,2))
            # nmg_repr_list = []
            # for strt_idx in range(0,curr_seq_len-nmg_seq_len+1,2):
            #     curr_pos, curr_ori = geometry.get_ground_point(smpl_trans[strt_idx],p3d_rt.rotation_6d_to_matrix(smpl_orient[strt_idx]))
            #     curr_tfm = to_homogeneous(curr_ori,curr_pos)
            #     curr_root_pose = to_homogeneous(p3d_rt.rotation_6d_to_matrix(smpl_orient[strt_idx:strt_idx+nmg_seq_len]),smpl_trans[strt_idx:strt_idx+25])
            #     canonical_root_pose = torch.matmul(torch.inverse(curr_tfm),curr_root_pose)
            #     canonical_smpl_motion = torch.cat([canonical_root_pose[:,:3,3],
            #                             p3d_rt.matrix_to_axis_angle(canonical_root_pose[:,:3,:3]),
            #                             smpl_motion[strt_idx:strt_idx+nmg_seq_len,6:]],dim=1)
            #     nmg_repr = (smpl2nmg(canonical_smpl_motion,smpl).reshape(-1,25,22*9) - mvae_mean)/mvae_std
            #     nmg_repr_list.append(nmg_repr)
            
            nmg_repr = torch.cat(nmg_repr_list,dim=0)
            smpl_motion_latent = mvae_model.encode(nmg_repr)[:,0]
            # nmg_repr = (smpl2nmg(smpl_motion,smpl).reshape(-1,curr_seq_len,22*9) - mvae_mean)/mvae_std
            # smpl_motion_latent = mvae_model.encode(nmg_repr)[:,0]

            # SMPL fwd pass
            smpl_out = smpl.forward(root_orient = smpl_motion[:,3:6],
                                        pose_body = smpl_motion[:,6:],
                                        trans = smpl_motion[:,:3],
                                        betas = smpl_shape.unsqueeze(1).expand(-1,curr_seq_len,-1).reshape(-1,smpl_shape.shape[-1]))
            # smpl_out = smpl2.forward(global_orient = p3d_rt.rotation_6d_to_matrix(smpl_orient).unsqueeze(1),
            #                             body_pose = p3d_rt.rotation_6d_to_matrix(smpl_art_motion),
            #                             transl = smpl_motion[:,:3],
            #                             betas = smpl_shape.unsqueeze(1).expand(-1,seq_len,-1).reshape(-1,smpl_shape.shape[-1]),pose2rot=False)
            
            j3ds = smpl_out.Jtr[:,:22,:]
            
            # camera extrinsics
            cam_orient_expanded = torch.cat([x.repeat(1,curr_seq_len,1) if x.shape[1]==1 else x for x in cam_orient],dim=0)
            cam_position_expanded = torch.cat([x.repeat(1,curr_seq_len,1) if x.shape[1]==1 else x for x in cam_position],dim=0)
            cam_ext_temp = torch.cat([p3d_rt.rotation_6d_to_matrix(cam_orient_expanded),cam_position_expanded.unsqueeze(3)],dim=3)
            cam_ext = torch.cat([cam_ext_temp,torch.tensor([0,0,0,1]).type_as(cam_ext_temp).repeat(num_cams,curr_seq_len,1,1)],dim=2)

            # camera extrinsics
            cam_poses = torch.inverse(cam_ext)

            # camera projection
            proj_j3ds = torch.stack([perspective_projection(j3ds,
                            cam_ext[i,:,:3,:3].reshape(-1,3,3),
                            cam_ext[i,:,:3,3].reshape(-1,3),
                            cam_intr[i].unsqueeze(0).expand(curr_seq_len,-1,-1).reshape(-1,3,3)).reshape(curr_seq_len,-1,2) for i in range(num_cams)])


            ####################### Losses #######################

            # reprojection loss         
            if stage == 0:
                loss_2d = (j2ds[:,:,:22,2]*(((proj_j3ds[:,:,:22] - j2ds[:,:,:22,:2])**2).sum(dim=3))).sum(dim=0).mean()
            elif stage == 1:
                # take only first frame
                loss_2d = (j2ds[:,:,:22,2]*(((proj_j3ds[:,:,:22] - j2ds[:,:,:22,:2])**2).sum(dim=3))).sum(dim=0).mean()
                # loss_2d = (j2ds[:,:,idcs,2]*((gmcclure(proj_j3ds[:,:,idcs] - j2ds[:,:,idcs,:2],config["gmcclure_sigma"])).sum(dim=3))).sum(dim=0).mean()
            else:
                loss_2d = (j2ds[:,:,:22,2]*(((proj_j3ds[:,:,:22] - j2ds[:,:,:22,:2])**2).sum(dim=3))).sum(dim=0).mean()

            # latent space regularization
            loss_z = (smpl_motion_latent**2).mean(dim=-1).sum()

            # j3d smooth loss
            loss_j3d_smooth = (((j3ds[1:] - j3ds[:-1])**2).sum(dim=2)).sum()

            # smooth camera motions
            # loss_cams_orient = ((cam_orient_expanded[:,1:] - cam_orient_expanded[:,:-1])**2).mean()
            # loss_cams_position = ((cam_position_expanded[:,1:] - cam_position_expanded[:,:-1])**2).mean()
            loss_cams_orient = ((cam_poses[:,1:,:3,:3] - cam_poses[:,:-1,:3,:3])**2).mean()
            loss_cams_position = ((cam_poses[:,1:,:3,3] - cam_poses[:,:-1,:3,3])**2).mean()

            # shape regularization loss
            loss_betas = (smpl_shape*smpl_shape).mean()

            # ground penetration loss
            loss_human_gp = torch.nn.functional.relu(-j3ds[:,:,2]).mean()

            # camera ground penetration loss
            loss_cam_gp = torch.nn.functional.relu(-cam_poses[:,:,2,3]).mean()

            # loss vposer
            loss_vp = (smpl_art_motion_vp_latent*smpl_art_motion_vp_latent).mean()

            # loss smpl in front
            loss_smpl_in_front = torch.nn.functional.relu(-cam_ext[:,:,2,3]).mean()

            # total loss
            loss = loss_2d_weight[stage] * loss_2d + \
                    loss_z_weight[stage] * loss_z + \
                        loss_cams_orient_weight[stage] * loss_cams_orient + \
                            loss_cams_position_weight[stage] * loss_cams_position + \
                            loss_betas_weight[stage] * loss_betas + \
                                float(loss_human_gp_weight[stage]) * loss_human_gp + \
                                    float(loss_cam_gp_weight[stage]) * loss_cam_gp + \
                                        loss_vp_weight[stage] * loss_vp + \
                                            loss_smpl_in_front_weight[stage] * loss_smpl_in_front + \
                                                loss_j3d_smooth_weight[stage] * loss_j3d_smooth

            # Viz

            # zero grad optimizer
            optim.zero_grad()
            
            # backward
            loss.backward()

            # optim step
            optim.step()

            # schedule lr
            # sched.step(loss)

            # loss dict
            loss_dict = {"loss":loss.item(),
                            "loss_2d":loss_2d.item(),
                            "loss_z":loss_z.item(),
                            "loss_vp":loss_vp.item(),
                            "loss_j3d":loss_j3d_smooth.item(),
                            "loss_cams_orient":loss_cams_orient.item(),
                            "loss_cams_position":loss_cams_position.item(),
                            "loss_betas":loss_betas.item(),
                            "loss_human_gp":loss_human_gp.item(),
                            "loss_cam_gp":loss_cam_gp.item(),
                            "loss_smpl_front":loss_smpl_in_front.item()}

            # print loss
            t.set_postfix(loss_dict)

            # track losses
            # for k,v in loss_dict.items():
            #     writer.add_scalar(k,v,global_step=i)
    #         if i == 0 or i == iters[0] or i == iters[1] :
    #             import ipdb;ipdb.set_trace()
    # import ipdb;ipdb.set_trace()
    return {"cam_orient":[p3d_rt.rotation_6d_to_matrix(x).clone().detach().cpu() for x in cam_orient], "cam_position": [x.clone().detach().cpu() for x in cam_position],
                        "smpl_trans":smpl_trans.clone().detach().cpu(), "proj_j3ds":proj_j3ds.clone().detach().cpu(),
                        "smpl_orient": p3d_rt.rotation_6d_to_matrix(smpl_orient).cpu(), "smpl_shape": smpl_shape.clone().detach().cpu(), 
                        "smpl_art_motion_vp_latent":smpl_art_motion_vp_latent.clone().detach().cpu(), "j2ds":j2ds.clone().detach().cpu(),
                        "full_im_paths": params["full_im_paths"]}



def stitch(prev_stage_dicts,stitch_num):
    
    new_stage_dicts = []

    with torch.no_grad():

        for n_dict in range(0,len(prev_stage_dicts),stitch_num):
            prev_dict = prev_stage_dicts[n_dict]
            cam_orient = [x.clone().detach() for x in prev_dict["cam_orient"]]
            cam_position = [x.clone().detach() for x in prev_dict["cam_position"]]
            smpl_trans = prev_dict["smpl_trans"].clone().detach()
            smpl_shape = prev_dict["smpl_shape"].clone().detach()
            smpl_orient = prev_dict["smpl_orient"].clone().detach()
            smpl_art_motion_vp_latent = prev_dict["smpl_art_motion_vp_latent"].clone().detach()
            last_smpl_orient = prev_dict["smpl_orient"].clone().detach()
            last_smpl_trans = prev_dict["smpl_trans"].clone().detach()
            j2ds = prev_dict["j2ds"].clone().detach()
            full_im_paths = prev_dict["full_im_paths"]
            
            for i in range(n_dict+1,n_dict+stitch_num):
                if i >= len(prev_stage_dicts):
                    break
                prev_dict = prev_stage_dicts[i]
                pos, ori = geometry.get_ground_point(last_smpl_trans[-1],last_smpl_orient[-1])
                prev_tfm = to_homogeneous(ori,pos)
                curr_pos, curr_ori = geometry.get_ground_point(prev_dict["smpl_trans"][overlap-1],prev_dict["smpl_orient"][overlap-1])
                curr_tfm = to_homogeneous(curr_ori,curr_pos)
                tfm = torch.matmul(prev_tfm,torch.inverse(curr_tfm))
                curr_smpl = to_homogeneous(prev_dict["smpl_orient"],prev_dict["smpl_trans"])
                tfmd_smpl = torch.matmul(tfm,curr_smpl)
                
                smpl_orient = torch.cat([smpl_orient, tfmd_smpl[overlap-1:,:3,:3]])
                smpl_trans = torch.cat([smpl_trans,tfmd_smpl[overlap-1:,:3,3]]).clone().detach()

                smpl_art_motion_vp_latent = torch.cat([smpl_art_motion_vp_latent,prev_dict["smpl_art_motion_vp_latent"][overlap-1:]])

                updated_cam_orient = [torch.matmul(tfm[:,:3,:3],x[0,:,:3,:3]) for x in prev_dict["cam_orient"]]
                updated_cam_position = [torch.matmul(tfm[:,:3,:3],x[0,:].unsqueeze(-1)).squeeze(-1) + tfm[0,:3,3] for x in prev_dict["cam_position"]]
                
                for j in range(len(cam_orient)):
                    if cam_orient[j].shape[1] != 1:
                        cam_orient[j] = torch.cat([cam_orient[j], updated_cam_orient[j].unsqueeze(0)[:,overlap-1:]],dim=1)
                        cam_position[j] = torch.cat([cam_position[j], updated_cam_position[j].unsqueeze(0)[:,overlap-1:]],dim=1)

                # save last smpl orient
                last_smpl_orient = tfmd_smpl[:,:3,:3].clone().detach()
                last_smpl_trans = tfmd_smpl[:,:3,3].clone().detach()

                j2ds = torch.cat([j2ds,prev_dict["j2ds"].clone().detach()[:,overlap-1:]],dim=1)

                for k in range(len(full_im_paths)):
                    full_im_paths[k] = full_im_paths[k] + prev_dict["full_im_paths"][k][overlap-1:]
                

            new_stage_dicts.append({"cam_orient":cam_orient, "cam_position": cam_position,
                            "smpl_trans":smpl_trans.clone().detach(), 
                            "smpl_orient": smpl_orient, "smpl_shape": smpl_shape, 
                            "smpl_art_motion_vp_latent":smpl_art_motion_vp_latent.clone().detach(), "j2ds":j2ds.clone().detach(),
                            "full_im_paths": full_im_paths})

    return new_stage_dicts

            

def save_results(stage_dict,stage,seq_start, prefix="", viz=False):

    os.makedirs("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/stage_{:02d}/seq_start_{:05d}".format(config["trial_name"],
                                seq_no,stage,seq_start),exist_ok=True)

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

    if viz:
        for cam in tqdm(range(num_cams)):
            os.makedirs("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/stage_{:02d}/seq_start_{:05d}/cam_{:02d}".format(config["trial_name"],
                                    seq_no,stage,seq_start,cam),exist_ok=True)
            for s in range(curr_seq_len):
                im = cv2.imread(full_images_paths[cam][s])[:im_res[cam][1],:im_res[cam][0],::-1]/255.
                temp_intr = cam_intr[cam].clone().detach().cpu().numpy()
                temp_v = smpl_out_v.view(curr_seq_len,6890,3)[s].clone().detach().cpu().numpy()
                
                rend_ims = renderer[cam](temp_v,
                                    cam_ext[cam,s,:3,3].clone().detach().cpu().numpy(),
                                    cam_ext[cam,s,:3,:3].unsqueeze(0).clone().detach().cpu().numpy(),
                                    im,intr=temp_intr,
                                    faces=smpl.f.clone().detach().cpu().numpy())
                for joint in range(j2ds.shape[2]-2):
                    rend_ims = rend_ims*255
                    cv2.circle(rend_ims,(int(j2ds[cam,s,joint,0]),
                                int(j2ds[cam,s,joint,1])),10,(255,255,255),-1)
                    cv2.circle(rend_ims,(int(stage_dict["proj_j3ds"][cam,s,joint,0]),
                                int(stage_dict["proj_j3ds"][cam,s,joint,1])),5,(0,0,0),-1)
                    rend_ims = rend_ims/255
                
                cv2.imwrite("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/stage_{:02d}/seq_start_{:05d}/cam_{:02d}/{:05d}.png".format(config["trial_name"],
                                    seq_no,stage,seq_start,cam,seq_start+s),rend_ims[::viz_dwnsample,::viz_dwnsample,::-1]*255)

    if prefix == "":
        pkl.dump(stage_dict,open("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/stage_{:02d}/{}_seq_start_{:05d}.pkl".format(config["trial_name"],
                                seq_no,stage,prefix,seq_start),"wb"))

    np.savez("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/stage_{:02d}/{}_seq_start_{:05d}".format(config["trial_name"],
                                seq_no,stage,prefix,seq_start),
        verts=smpl_out_v.clone().detach().cpu().numpy(),
        cam_trans=cam_poses[:,:,:3,3].clone().detach().cpu().numpy(),
        cam_rots=p3d_rt.matrix_to_quaternion(cam_poses[:,:,:3,:3]).clone().detach().cpu().numpy())




if __name__ == "__main__":

    if resume:
        print("\n Resuming from checkpoint \n")
        # get existsing stages
        stage = sorted([int(x.split("_")[-1]) for x in stage_dirs])[-1]
        fitter_res_stage = []
        for stage_dir in stage_dirs:
            fitter_res_stage.append([pkl.load(open(x,"rb")) for x in sorted(glob.glob(os.path.join(stage_dir,"*.pkl")))])
    else:
        stage = 0
        fitter_res_stage = []
        iterations = config["n_optim_iters"][0]
        pare_init_state = load_batch_wth_pare_init()

        # optimize(pare_init_state[0],device,iterations)
        res_stage = Parallel(n_jobs=-1)(delayed(optimize)(pare_init_state[i],device,
                        iterations) for i in tqdm(range(len(pare_init_state))))

        # save results
        for i in tqdm(range(len(pare_init_state))):
            print("\n saving results \n")
            save_results(res_stage[i],stage,big_seq_start + i*(init_seq_len-overlap),viz=True)
        # Parallel(n_jobs=-1)(delayed(save_results)(res_stage[i],stage,big_seq_start + i*(init_seq_len-overlap)) for i in tqdm(range(len(pare_init_state))))
        

        fitter_res_stage.append(res_stage)

        print("\n stage length {} \n".format(len(res_stage)))
        

    if config["stitch_len"] == -1:
        stitch_len = len(fitter_res_stage[-1])
    else:
        stitch_len = config["stitch_len"]
    stitched_res = stitch(fitter_res_stage[-1],stitch_len)

    for i in range(len(stitched_res)):
        print("\n saving stitched results \n")
        save_results(stitched_res[i],stage,i,prefix="stitched")

    stage += 1
    
    print("\n stitched length {} \n".format(len(stitched_res)))

    iterations = config["n_optim_iters"][1]
    while len(fitter_res_stage[-1]) != 1:
        # if len(stitched_res) == 1:
        #     print("device is swtiched to CUDA")
        #     device = "cuda"
            # optimize(stitched_res[-1],device,iterations)
        res_stage = Parallel(n_jobs=-1)(delayed(optimize)(stitched_res[i],device,iterations) for i in tqdm(range(len(stitched_res))))

        for i in tqdm(range(len(res_stage))):
            print("\n saving results \n")
            save_results(res_stage[i],stage,big_seq_start + i*(stitched_res[i]["j2ds"].shape[1]-overlap))
        # Parallel(n_jobs=-1)(delayed(save_results)(res_stage[i],stage,big_seq_start + i*(stitched_res[i]["j2ds"].shape[1]-overlap)) for i in tqdm(range(len(stitched_res))))
        
        fitter_res_stage.append(res_stage)
        stitched_res = stitch(fitter_res_stage[-1],config["stitch_len"])
        if len(stitched_res)==0:
            stitched_res = stitch(fitter_res_stage[-1],len(fitter_res_stage[-1]))

        for i in range(len(stitched_res)):
            print("\n saving stitched results \n")
            save_results(stitched_res[i],stage,i,prefix="stitched")
        # Parallel(n_jobs=-1)(delayed(save_results)(res_stage[i],stage,i) for i in tqdm(range(len(stitched_res))))
        
        print("\n stage length {} \n".format(len(res_stage)))
        print("\n stitched length {} \n".format(len(stitched_res)))
        print("\n")

        stage = stage+1

        # restart job
        # if os.uname()[1].lower() != "ps106":
        #     print("\n Exiting to get restarted again \n")
        #     sys.exit(3)
        
    print("\n saving results \n")
    save_results(res_stage[0],stage,big_seq_start,viz=True)

