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
from mcms.utils.utils import to_homogeneous
import sys
import glob
import pickle as pkl


res_dir = sys.argv[1]

# load results
final_stage_res = glob.glob(ospj(sorted(glob.glob(os.path.join(res_dir,"stage_*")))[-1],"_*.pkl"))
if len(final_stage_res) > 1:
    import ipdb;ipdb.set_trace()

res = pkl.load(open(final_stage_res[0],"rb"))


# Params
config = yaml.safe_load(open(ospj(res_dir,"config.yml")))
batch_size = config["batch_size"]
loss_2d_weight = config["loss_2d_weight"]
loss_z_weight = config["loss_z_weight"]
loss_cams_weight = config["loss_cams_weight"]
loss_betas_weight = config["loss_betas_weight"]
n_optim_iters = config["n_optim_iters"]
loss_human_gp_weight = config["loss_human_gp_weight"]
loss_cam_gp_weight = config["loss_cam_gp_weight"]
loss_vp_weight = config["loss_vp_weight"]
loss_smpl_in_front_weight = config["loss_smpl_in_front_weight"]
lr = config["lr"]
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


# gt params
gt_trans = batch["gt_trans"].float().to(device)
gt_global_orient = batch["gt_global_orient"].float().to(device)
gt_betas = batch["gt_betas"].float().to(device)
gt_body_pose = batch["gt_body_pose"].float().to(device)
gt_cam_poses = torch.inverse(torch.from_numpy(batch["cam_extr"]).float().to(device))
gt_cam_rots = gt_cam_poses[:,:3,:3]
gt_cam_trans = gt_cam_poses[:,:3,3]
gt_j3d_zero_trans = smpl.forward(root_orient = p3d_rt.matrix_to_axis_angle(gt_global_orient).squeeze(1),
                                            pose_body = p3d_rt.matrix_to_axis_angle(gt_body_pose).reshape(-1,69)[:,:63],
                                            trans = torch.zeros(gt_trans.shape[0],3).type_as(gt_trans),
                                            betas = gt_betas).Jtr[:,:22]
gt_j3d_zero_trans_orient = smpl.forward(root_orient = torch.zeros(gt_global_orient.shape[0],3).type_as(gt_global_orient),
                                            pose_body = p3d_rt.matrix_to_axis_angle(gt_body_pose).reshape(-1,69)[:,:63],
                                            trans = torch.zeros(gt_trans.shape[0],3).type_as(gt_trans),
                                            betas = gt_betas).Jtr[:,:22]

# load results
res_trans = res["smpl_trans"].float().to(device)
res_global_orient = res["smpl_orient"].float().to(device)
res_smpl_root_pose = to_homogeneous(res_global_orient,res_trans)
res_body_pose = vp_model.decode(res["smpl_art_motion_vp_latent"])["pose_body"].float().to(device).reshape(seq_len,63)
res_betas = res["smpl_shape"].float().to(device)
res_cam_rots = torch.cat([x.repeat(1,seq_len,1,1) if x.shape[1] == 1 else x for x in res["cam_orient"]]).float().to(device)
res_cam_trans = torch.cat([x.repeat(1,seq_len,1) if x.shape[1] == 1 else x for x in res["cam_position"]]).float().to(device)
res_cam_poses = to_homogeneous(res_cam_rots,res_cam_trans)
res_origin_wrt_cam0 = torch.inverse(res_cam_poses[0:1,0:1])
res_cam_poses_wrt_cam0 = torch.matmul(res_origin_wrt_cam0,res_cam_poses)
res_smpl_root_pose_wrt_cam0 = torch.matmul(res_origin_wrt_cam0,res_smpl_root_pose)
res_j3d_zero_trans = smpl.forward(root_orient = p3d_rt.matrix_to_axis_angle(res_global_orient),
                                            pose_body = res_body_pose,
                                            trans = torch.zeros(res_trans.shape[0],3).type_as(res_trans),
                                            betas = res_betas.repeat(res_trans.shape[0],1)).Jtr[:,:22]
res_j3d_zero_trans_orient = smpl.forward(root_orient = torch.zeros(res_global_orient.shape[0],3).type_as(res_global_orient),
                                            pose_body = res_body_pose,
                                            trans = torch.zeros(res_trans.shape[0],3).type_as(res_trans),
                                            betas = res_betas.repeat(res_trans.shape[0],1)).Jtr[:,:22]

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