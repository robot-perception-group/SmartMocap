from concurrent.futures import process
from uuid import RESERVED_MICROSOFT
from matplotlib.style import available
import numpy as np
import os
from os.path import join as ospj
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.tools.model_loader import load_model
from pytorch3d.transforms import rotation_conversions as p3d_rt
from pytorch3d import transforms as p3dt
from nmg.models import nmg
from mcms.utils.utils import nmg2smpl, smpl2nmg
from mcms.utils import geometry
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
humor_res_dir = sys.argv[2]
start_offset = int(sys.argv[3])

# load results
stages = [x for x in sorted(glob.glob(os.path.join(res_dir,"stage_*"))) if x.split("_")[-1] != "99"]
final_stage_res = glob.glob(ospj(stages[-1],"_*.pkl"))

if len(final_stage_res) > 1:
    import ipdb;ipdb.set_trace()

res = pkl.load(open(final_stage_res[0],"rb"))


# Params
config = yaml.safe_load(open(ospj(res_dir,"config.yml")))
batch_size = config["batch_size"]
nmg_seq_len = config["nmg_seq_len"]
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

# MVAE
nmg_hparams = yaml.safe_load(open("/".join(hparams["train_motion_vae_ckpt_path"].split("/")[:-2])+"/hparams.yaml"))
mvae_model = nmg.nmg.load_from_checkpoint(hparams["train_motion_vae_ckpt_path"],map_location=device).to(device)
mean_std = np.load(hparams["model_mvae_mean_std_path"])
mvae_mean = torch.from_numpy(mean_std["mean"]).float().to(device)
mvae_std = torch.from_numpy(mean_std["std"]).float().to(device)

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

if dset.lower() != "rich":
    import ipdb;ipdb.set_trace()


seq_start = big_seq_start
seq_len = res["smpl_trans"].shape[0]
if dset.lower() == "h36m":
    # get batch
    batch = ds.__getitem__(seq_no,seq_start,seq_len)
else:
    batch = ds.__getitem__(seq_start,seq_len)



# load results
res = np.load(humor_res_dir)
res_wrt_cam = np.load(ospj("/".join(humor_res_dir.split("/")[:-1]),"stage3_results.npz"))
cam_wrt_res_root = torch.inverse(to_homogeneous(p3d_rt.axis_angle_to_matrix(
                    torch.from_numpy(res_wrt_cam["root_orient"][start_offset+config["big_seq_start"]:batch["gt_body_pose"].shape[0]+config["big_seq_start"]])),
                    torch.from_numpy(res_wrt_cam["trans"][start_offset+config["big_seq_start"]:batch["gt_body_pose"].shape[0]+config["big_seq_start"]])).float().to(device))
res_trans = torch.from_numpy(res["trans"]).float().to(device)[start_offset+config["big_seq_start"]:batch["gt_body_pose"].shape[0]+config["big_seq_start"]]
res_global_orient = torch.from_numpy(res["root_orient"]).float().to(device)[start_offset+config["big_seq_start"]:batch["gt_body_pose"].shape[0]+config["big_seq_start"]]
res_body_pose = torch.from_numpy(res["pose_body"]).float().to(device)[start_offset+config["big_seq_start"]:batch["gt_body_pose"].shape[0]+config["big_seq_start"]]

res_betas = torch.from_numpy(res["betas"][:,:10]).float().to(device)[start_offset+config["big_seq_start"]:batch["gt_body_pose"].shape[0]+config["big_seq_start"]]
res_trans_zero_corr = res_trans.detach().clone()
res_trans_zero_corr[:,:2] = res_trans_zero_corr[:,:2] - res_trans_zero_corr[0,:2]
res_norm_poses,res_norm_tfm = get_norm_poses(torch.cat([res_global_orient,res_body_pose],dim=1),res_trans_zero_corr)
res_root_wrt_origin = to_homogeneous(p3d_rt.axis_angle_to_matrix(res_norm_poses[:,3:6]),res_norm_poses[:,:3]).float().to(device)
# use only first frame
res_cam_poses = torch.matmul(res_root_wrt_origin,cam_wrt_res_root).unsqueeze(0)
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


available_gt_cam_poses = torch.stack([gt_cam_poses[x] if x < gt_cam_poses.shape[0] else gt_cam_poses[0] for x in config["cams_used"]])
# np.savez(ospj(res_dir,"res_strt_off_"+str(start_offset)),verts=res_smplout.v.detach().cpu().numpy(),
#             cam_trans=res_cam_poses[:,:,:3,3].detach().cpu().numpy(),
#             cam_rots=p3d_rt.matrix_to_quaternion(res_cam_poses[:,:,:3,:3]).detach().cpu().numpy())
np.savez(ospj(res_dir,"gt_and_humor_res_strt_off_"+str(start_offset)),verts=torch.stack([gt_smplout.v,res_smplout.v]).detach().cpu().numpy(),
            cam_trans=torch.stack([available_gt_cam_poses[:,:3,3].unsqueeze(1).repeat([1,gt_smplout.v.shape[0],1]),
            res_cam_poses[:,:,:3,3]]).detach().cpu().numpy(),
            cam_rots=p3d_rt.matrix_to_quaternion(torch.stack([available_gt_cam_poses[:,:3,:3].unsqueeze(1).repeat([1,gt_smplout.v.shape[0],1,1]),
            res_cam_poses[:,:,:3,:3]])).detach().cpu().numpy())


# metrics
################## assuming rich dataset $$$$$$$$$$$$$$$$$$$$$$$$
print("\n !!!!!!!!!!!!!!!! Assuming moving camera is number 7 !!!!!!!!!!!!!!!!!!!!")
if 7 in config["cams_used"]:
    cam_position_error = ((available_gt_cam_poses[:-1, :3, 3] - res_cam_poses[:-1, 0, :3, 3])**2).sum(dim=-1).sqrt().detach().cpu().numpy()
    cam_orientation_error = p3dt.so3_relative_angle(available_gt_cam_poses[:-1,:3,:3],res_cam_poses[:-1,0,:3,:3])
else:
    cam_position_error = ((available_gt_cam_poses[:, :3, 3] - res_cam_poses[:, 0, :3, 3])**2).sum(dim=-1).sqrt().detach().cpu().numpy()
    cam_orientation_error = p3dt.so3_relative_angle(available_gt_cam_poses[:,:3,:3],res_cam_poses[:,0,:3,:3])
    
smpl_position_error = ((norm_poses[:,:3] - res_norm_poses[:,:3])**2).sum(dim=-1).sqrt().detach().cpu().numpy()
smpl_orientation_error = p3dt.so3_relative_angle(p3d_rt.axis_angle_to_matrix(norm_poses[:,3:6]),p3d_rt.axis_angle_to_matrix(res_norm_poses[:,3:6]))
smpl_MPJPE_error = ((gt_smplout.Jtr[:,:22] - res_smplout.Jtr[:,:22])**2).sum(dim=-1).sqrt().mean(dim=0).detach().cpu().numpy()
smpl_MPJPE_zero_trans = ((smpl.forward(root_orient = norm_poses[:,3:6],
                            pose_body = norm_poses[:,6:69],
                            trans = torch.zeros(norm_poses.shape[0],3).type_as(norm_poses),
                            betas = gt_betas[start_offset:]).Jtr[:,:22] - smpl.forward(root_orient = res_norm_poses[:,3:6],
                                                                    pose_body = res_norm_poses[:,6:69],
                                                                    trans = torch.zeros(res_norm_poses.shape[0],3).type_as(res_norm_poses),
                                                                    betas = res_betas).Jtr[:,:22])**2).sum(dim=-1).sqrt().mean(dim=0).detach().cpu().numpy()
smpl_MPJPE_zero_trans_orient_beta = ((smpl.forward(root_orient = torch.zeros(norm_poses.shape[0],3).type_as(norm_poses),
                                pose_body = norm_poses[:,6:69],
                                trans = torch.zeros(norm_poses.shape[0],3).type_as(norm_poses),
                                betas = torch.zeros(norm_poses.shape[0],10).type_as(norm_poses)).Jtr[:,:22] - 
                                smpl.forward(root_orient = torch.zeros(res_norm_poses.shape[0],3).type_as(res_norm_poses),
                                            pose_body = res_norm_poses[:,6:69],
                                            trans = torch.zeros(res_norm_poses.shape[0],3).type_as(res_norm_poses),
                                            betas = torch.zeros(res_norm_poses.shape[0],10).type_as(res_norm_poses)).Jtr[:,:22])**2).sum(dim=-1).sqrt().mean(dim=0).detach().cpu().numpy()

smpl_MPVPE_zero_trans_orient_theta = ((smpl.forward(root_orient = torch.zeros(norm_poses.shape[0],3).type_as(norm_poses),
                                pose_body = torch.zeros(norm_poses.shape[0],63).type_as(norm_poses),
                                trans = torch.zeros(norm_poses.shape[0],3).type_as(norm_poses),
                                betas = gt_betas[start_offset:]).v- 
                                smpl.forward(root_orient = torch.zeros(res_norm_poses.shape[0],3).type_as(res_norm_poses),
                                            pose_body = torch.zeros(res_norm_poses.shape[0],63).type_as(res_norm_poses),
                                            trans = torch.zeros(res_norm_poses.shape[0],3).type_as(res_norm_poses),
                                            betas = res_betas).v)**2).sum(dim=-1).sqrt().mean(dim=0).detach().cpu().numpy()

res_dict = {"CPE": cam_position_error.tolist(), "COE": cam_orientation_error.tolist(), "SMPL_PE": smpl_position_error.tolist(), 
            "SMPL_OE": smpl_orientation_error.tolist(), 
            "SMPL_MOE": smpl_orientation_error.mean().tolist(),
            "SMPL_MOE_std": smpl_orientation_error.std().tolist(),
            "SMPL_PJPE":smpl_MPJPE_error.tolist(),"SMPL_PJPE_tau0":smpl_MPJPE_zero_trans.tolist(),
            "SMPL_PJPE_tau0_phi0_beta0":smpl_MPJPE_zero_trans_orient_beta.tolist(),
            "CPE_mean": cam_position_error.mean().tolist(),
            "CPE_std": cam_position_error.std().tolist(),
            "COE_mean": cam_orientation_error.mean().tolist(),
            "COE_std": cam_orientation_error.std().tolist(),
            "SMPL_MPE":smpl_position_error.mean().tolist(),
            "SMPL_MPE_std":smpl_position_error.std().tolist(),
            "SMPL_MPJPE":smpl_MPJPE_error.mean().tolist(),
            "SMPL_MPJPE_std":smpl_MPJPE_error.std().tolist(),
            "SMPL_MPJPE_tau0":smpl_MPJPE_zero_trans.mean().tolist(),
            "SMPL_MPJPE_tau0_std":smpl_MPJPE_zero_trans.std().tolist(),
            "SMPL_MPJPE_tau0_phi0_beta0":smpl_MPJPE_zero_trans_orient_beta.mean().tolist(),
            "SMPL_MPJPE_tau0_phi0_beta0_std":smpl_MPJPE_zero_trans_orient_beta.std().tolist(),
            "SMPL_PVPE_tau0_phi0_beta0":smpl_MPVPE_zero_trans_orient_theta.tolist(), 
            "SMPL_MPVPE_tau0_phi0_beta0":smpl_MPVPE_zero_trans_orient_theta.mean().tolist(),
            "SMPL_MPVPE_tau0_phi0_beta0_std":smpl_MPVPE_zero_trans_orient_theta.std().tolist()}

yaml.safe_dump(res_dict,open(ospj(res_dir,"gt_and_humor_res_strt_off_"+str(start_offset)+".yml"),"w"))


# import ipdb;ipdb.set_trace()

# import matplotlib.pyplot as plt
# plt.plot(smpl_MPJPE_zero_trans)
# plt.plot(smpl_MPJPE_zero_trans_orient)
# plt.plot(smpl_position_error)