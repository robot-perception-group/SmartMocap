from concurrent.futures import process
import numpy as np
import os
from os.path import join as ospj
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.tools.model_loader import load_model
from pytorch3d.transforms import rotation_conversions as p3d_rt
import yaml
import torch
import cv2
from mcms.dsets import h36m, copenet_real, rich
from savitr_pe.datasets import savitr_dataset
from mcms.utils.utils import proper_smpl_fwd, to_homogeneous, get_norm_poses
import sys
import glob
import pickle as pkl
import json
import trimesh


res_dir = sys.argv[1]

# load results
final_stage_res = glob.glob(ospj(sorted(glob.glob(os.path.join(res_dir,"stage_*")))[-1],"_*.pkl"))
if len(final_stage_res) > 1:
    import ipdb;ipdb.set_trace()

res = pkl.load(open(final_stage_res[0],"rb"))


# Params
config = yaml.safe_load(open(ospj(res_dir,"config.yml")))
batch_size = config["batch_size"]
n_optim_iters = config["n_optim_iters"]
dset = config["dset"]
seq_no = config["seq_no"]
big_seq_start = config["big_seq_start"]
big_seq_end = config["big_seq_end"]
overlap = config["overlap"]
hparams = yaml.safe_load(open("/".join(config["mo_vae_ckpt_path"].split("/")[:-2])+"/hparams.yaml","r"))
device = torch.device(config["device"])

# load smpl model
smpl = BodyModel(bm_fname=hparams["model_smpl_neutral_path"]).to(device)

# vposer model
vp_model = load_model(hparams["model_vposer_path"], model_code=VPoser,remove_words_in_model_weights="vp_model.")[0].to(device)
vp_model.eval()

# Dataloader
if dset.lower() == "h36m":
    ds = h36m.h36m(hparams,used_cams=config["cams_used"])
    fps_scl = 2
    viz_dwnsample = 1
    # renderer
    im_res = [[1000,1000],[1000,1000],[1000,1000],[1000,1000]]
elif dset.lower() == "copenet_real":
    hparams["data_datapath"] = "/home/nsaini/Datasets/copenet_data"
    ds = copenet_real.copenet_real(hparams,range(0,7000))
    fps_scl = 1
    viz_dwnsample = 1
    im_res=[[1920,1080],[1920,1080]]
elif dset.lower() == "savitr":
    hparams["data_datapath"] = config["data_path"]
    ds = savitr_dataset.savitr_dataset(hparams["data_datapath"],seq_len=25,used_cams=config["cams_used"])
    im_lists = ds.__getitem__(0,seq_len=1)["full_im_paths"]
    im_res = [[cv2.imread(i[0]).shape[1],cv2.imread(i[0]).shape[0]] for i in im_lists]
    fps_scl = 1
    viz_dwnsample = 1
elif dset.lower() == "rich":
    ds = rich.rich(config["data_path"],used_cams=config["cams_used"])
    im_res = [[4112,3008],[4112,3008],[4112,3008],[3008,4112],[4112,3008],[3008,4112],[4112,3008],[4112,3008]]
    fps_scl = 1
    viz_dwnsample = 1


seq_start = big_seq_start
seq_len = res["smpl_trans"].shape[0]
if dset.lower() == "h36m":
    # get batch
    batch = ds.__getitem__(seq_no,seq_start,seq_len)
else:
    batch = ds.__getitem__(seq_start,seq_len)

start_offset = 10


# load results
res_trans = res["smpl_trans"].float().to(device)[start_offset:]
res_global_orient = res["smpl_orient"].float().to(device)[start_offset:]
res_smpl_root_pose = to_homogeneous(res_global_orient,res_trans)
res_body_pose = vp_model.decode(res["smpl_art_motion_vp_latent"])["pose_body"].float().to(device).reshape(seq_len,63)[start_offset:]
res_betas = res["smpl_shape"].float().to(device)
res_cam_extr_rots = torch.cat([x.repeat(1,seq_len,1,1) if x.shape[1] == 1 else x for x in res["cam_orient"]]).float().to(device)
res_cam_extr_trans = torch.cat([x.repeat(1,seq_len,1) if x.shape[1] == 1 else x for x in res["cam_position"]]).float().to(device)
res_cam_poses = torch.inverse(to_homogeneous(res_cam_extr_rots,res_cam_extr_trans)[:,start_offset:])
res_trans_zero_corr = res_trans.detach().clone()
res_trans_zero_corr[:,:2] = res_trans_zero_corr[:,:2] - res_trans_zero_corr[0,:2]
res_norm_poses,res_norm_tfm = get_norm_poses(torch.cat([p3d_rt.matrix_to_axis_angle(res_global_orient),res_body_pose],dim=1),res_trans)
res_cam_poses[:,:,:2,3] = res_cam_poses[:,:,:2,3] - res_trans[0,:2]
res_cam_poses = torch.matmul(to_homogeneous(res_norm_tfm,torch.zeros(1,3)), res_cam_poses)
res_smplout = smpl.forward(root_orient=res_norm_poses[:,3:6],
                                    pose_body = res_norm_poses[:,6:69],
                                    trans = res_norm_poses[:,:3],
                                    betas = res_betas)

# gt params
ioi2scan_pose = json.load(open(ospj(config["data_path"],"cam2scan.json"),"r"))
ground_z_offset = torch.from_numpy(np.mean(trimesh.load(ospj(config["data_path"],"ground_mesh.ply"),process=False).vertices,axis=0)).float().to(device)
ioi2scan_orient = torch.from_numpy(np.array(ioi2scan_pose["R"])).float().to(device).T
ioi2scan_position = torch.from_numpy(np.array(ioi2scan_pose["t"])).float().to(device) - ground_z_offset
ioi2scan_pose = to_homogeneous(ioi2scan_orient,ioi2scan_position)
gt_betas = batch["gt_betas"].float().to(device)
gt_body_pose = batch["gt_body_pose"].float().to(device)
pelvis_cam0 = smpl.forward(betas=gt_betas).Jtr[:,0,:]
gt_root_pose = torch.matmul(ioi2scan_pose.unsqueeze(0),
                to_homogeneous(batch["gt_global_orient"].float().to(device).squeeze(1),batch["gt_trans"].float().to(device)+pelvis_cam0))
gt_root_pose[:,:3,3] -= pelvis_cam0
gt_trans = gt_root_pose[start_offset:,:3,3]
gt_trans[:,:2] = gt_trans[:,:2] - gt_trans[0,:2]
norm_poses,norm_tfm = get_norm_poses(p3d_rt.matrix_to_axis_angle(torch.cat([gt_root_pose[start_offset:,:3,:3].unsqueeze(1),
                        gt_body_pose[start_offset:]],dim=1)),
                        gt_trans)
gt_cam_poses = torch.matmul(to_homogeneous(norm_tfm,torch.zeros(1,3)),
                            torch.matmul(ioi2scan_pose.unsqueeze(0),
                            torch.inverse(torch.from_numpy(batch["cam_extr"]).float().to(device))))

gt_smplout = smpl.forward(root_orient=norm_poses[:,3:6],
                                    pose_body = norm_poses[:,6:69],
                                    trans = norm_poses[:,:3],
                                    betas = gt_betas[start_offset:])

np.savez(ospj(res_dir,"res_strt_off_"+str(start_offset)),verts=res_smplout.v.detach().cpu().numpy(),
            cam_trans=res_cam_poses[:,:,:3,3],
            cam_rots=p3d_rt.matrix_to_quaternion(res_cam_poses[:,:,:3,:3]))
np.savez(ospj(res_dir,"gt_and_res_strt_off_"+str(start_offset)),verts=torch.stack([gt_smplout.v,res_smplout.v]).detach().cpu().numpy(),
            cam_trans=torch.stack([gt_cam_poses[config["cams_used"],:3,3].unsqueeze(1).repeat([1,gt_smplout.v.shape[0],1]),res_cam_poses[:,:,:3,3]]),
            cam_rots=p3d_rt.matrix_to_quaternion(torch.stack([gt_cam_poses[config["cams_used"],:3,:3].unsqueeze(1).repeat([1,gt_smplout.v.shape[0],1,1]),res_cam_poses[:,:,:3,:3]])))

import ipdb;ipdb.set_trace()



# errors
cam_position_error = ((gt_cam_poses[1:, :3, 3] - res_cam_poses_wrt_cam0[1:-1, 0, :3, 3])**2).sum(dim=1).sqrt()
smpl_position_error = ((gt_trans - res_smpl_root_pose_wrt_cam0[0,:,:3,3])**2).sum(dim=1).sqrt()
smpl_MPJPE_zero_trans = ((gt_j3d_zero_trans - res_j3d_zero_trans)**2).sum(dim=2).sqrt().mean(dim=1)
smpl_MPJPE_zero_trans_orient = ((gt_j3d_zero_trans_orient - res_j3d_zero_trans_orient)**2).sum(dim=2).sqrt().mean(dim=1)

print("camera position error:")
print(cam_position_error.detach().cpu().numpy())
print("SMPL position error: {}".format(smpl_position_error.mean().detach().cpu().numpy()))
print("SMPL MPJPE zero translation: {}".format(smpl_MPJPE_zero_trans.mean().detach().cpu().numpy()))
print("SMPL MPJPE zero translation and orientation: {}".format(smpl_MPJPE_zero_trans_orient.mean().detach().cpu().numpy()))

import matplotlib.pyplot as plt
plt.plot(smpl_MPJPE_zero_trans.detach().cpu().numpy())
plt.plot(smpl_MPJPE_zero_trans_orient.detach().cpu().numpy())
plt.plot(smpl_position_error.detach().cpu().numpy())