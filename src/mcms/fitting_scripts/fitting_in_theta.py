import os
os.environ["PYOPENGL_PLATFORM"] = 'egl'
from mcms.models import mcms
import yaml
import torch
from pytorch3d.transforms import rotation_conversions as p3d_rt
import imageio
from torchvision.utils import make_grid

from mcms.dsets import h36m, copenet_real, rich
from savitr_pe.datasets import savitr_dataset
from torch.utils.data import DataLoader
from nmg.models import nmg
import numpy as np
from tqdm import tqdm, trange
from mcms.utils.utils import nmg2smpl, smpl2nmg
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.tools.model_loader import load_model
from mcms.utils.geometry import perspective_projection
from mcms.utils.renderer import Renderer
import cv2
from mcms.utils import geometry
from mcms.utils.utils import to_homogeneous, gmcclure
import sys



# Params
config = yaml.safe_load(open("/is/ps3/nsaini/projects/mcms/src/mcms/fitting_scripts/fit_config.yml"))
batch_size = config["batch_size"]
seq_len = config["seq_len"]
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


# SMPL
smpl = BodyModel(bm_fname=hparams["model_smpl_neutral_path"])

# Motion VAE
nmg_hparams = yaml.safe_load(open("/".join(hparams["train_motion_vae_ckpt_path"].split("/")[:-2])+"/hparams.yaml"))
mvae_model = nmg.nmg.load_from_checkpoint(hparams["train_motion_vae_ckpt_path"],nmg_hparams)
mean_std = np.load(hparams["model_mvae_mean_std_path"])
mvae_mean = torch.from_numpy(mean_std["mean"]).float()
mvae_std = torch.from_numpy(mean_std["std"]).float()

# vposer model
vp_model = load_model(hparams["model_vposer_path"], model_code=VPoser,remove_words_in_model_weights="vp_model.")[0]
vp_model.eval()

# Dataloader
if dset.lower() == "h36m":
    ds = h36m.h36m(hparams)
    fps_scl = 2
    # renderer
    im_res = [[1000,1000],[1000,1000],[1000,1000],[1000,1000]]
elif dset.lower() == "copenet_real":
    hparams["data_datapath"] = "/home/nsaini/Datasets/copenet_data"
    ds = copenet_real.copenet_real(hparams,range(0,7000))
    fps_scl = 1
    im_res=[[1920,1080],[1920,1080]]
elif dset.lower() == "savitr":
    hparams["data_datapath"] = config["data_path"]
    ds = savitr_dataset.savitr_dataset(hparams["data_datapath"],seq_len=25)
    im_lists = ds.__getitem__(0,seq_len=1)["full_im_paths"]
    im_res = [[cv2.imread(i[0]).shape[1],cv2.imread(i[0]).shape[0]] for i in im_lists]
    fps_scl = 1
elif dset.lower() == "rich":
    hparams["data_datapath"] = "/ps/project/datasets/AirCap_ICCV19/RICH_IPMAN/test/2021-06-15_Multi_IOI_ID_00186_Yoga1"
    ds = rich.rich(hparams["data_datapath"])
    im_res = [[4112,3008],[4112,3008],[4112,3008],[3008,4112],[4112,3008],[3008,4112],[4112,3008],[4112,3008]]
    fps_scl = 1
renderer = [Renderer(img_res=res) for res in im_res]
dl = DataLoader(ds, batch_size=batch_size,num_workers=0)


# Full fittings
full_cam_orient = []
full_cam_position = []
full_smpl_verts = []
full_smpl_shape = []
full_smpl_motion_latent = []
full_cam_ext = []
# overlay_gifs = []

# make dir for seq_no
if dset.lower() == "h36m":
    os.makedirs("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}".format(config["trial_name"],seq_no),exist_ok=True)
else:
    os.makedirs("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}".format(config["trial_name"]),exist_ok=True)



# Camera and SMPL params
# cam_position_staticcam = torch.tensor([0,0,5]).float().repeat(batch_size,num_cams,1,1).requires_grad_(True)
# 0,-5,-0.0039813 for actual poses (not extr)
# cam_orient_movingcam = p3d_rt.matrix_to_rotation_6d(p3d_rt.axis_angle_to_matrix(torch.tensor([3.14/2,0,0]).float())).repeat(batch_size,num_cams,seq_len,1).requires_grad_(True)
cam_position = []
cam_orient = []
for c in range(ds.num_cams):
    if c in config["position_changing_cams"]:
        cam_position.append(torch.tensor([0,2,5]).float().repeat(batch_size,1,seq_len,1).requires_grad_(True))
    else:
        cam_position.append(torch.tensor([0,2,5]).float().repeat(batch_size,1,1,1).requires_grad_(True))
    if c in config["orient_changing_cams"]:
        cam_orient.append(p3d_rt.matrix_to_rotation_6d(p3d_rt.axis_angle_to_matrix(torch.tensor([3.14/2,
                    0,0]).float())).repeat(batch_size,1,seq_len,1).requires_grad_(True))
    else:
        cam_orient.append(p3d_rt.matrix_to_rotation_6d(p3d_rt.axis_angle_to_matrix(torch.tensor([3.14/2,
                    0,0]).float())).repeat(batch_size,1,1,1).requires_grad_(True))
smpl_motion = torch.zeros(seq_len, 69).requires_grad_(True)
smpl_shape = torch.zeros(10).unsqueeze(0).requires_grad_(True)




# with trange(451,ds.data_lengths[seq_no]-50,50) as seq_t:
# with torch.autograd.set_detect_anomaly(True):
with trange(big_seq_start,big_seq_end,fps_scl*(seq_len-overlap)) as seq_t:
    for seq_start in seq_t:
        if dset.lower() == "h36m":
            # get batch
            batch = ds.__getitem__(seq_no,seq_start)
        else:
            batch = ds.__getitem__(seq_start)
        j2ds = batch["j2d"].float().unsqueeze(0)
        batch_size = j2ds.shape[0]
        num_cams = j2ds.shape[1]
        seq_len = j2ds.shape[2]

        # camera intrinsics
        if config["optimize_intr"]:
            focal_len = torch.tensor([3000 for _ in range(num_cams)]).float().requires_grad_(True)
        else:
            cam_intr = torch.from_numpy(batch["cam_intr"]).float().unsqueeze(0)

        # camera extrinsics
        cam_position = [cam_position[c].detach().requires_grad_(True) for c in range(num_cams)]
        cam_orient = [cam_orient[c].detach().requires_grad_(True) for c in range(num_cams)]
        smpl_motion = smpl_motion.detach().requires_grad_(True)
        smpl_shape = smpl_shape.detach().requires_grad_(True)
        
        ################# Optimizer #################
        if config["optimize_intr"]:
            optim = torch.optim.Adam(cam_orient + cam_position + [focal_len, smpl_motion, smpl_shape],lr=lr)
        else:
            optim = torch.optim.Adam(cam_orient + cam_position + [smpl_motion, smpl_shape],lr=lr)

        with trange(n_optim_iters) as t:
            for i in t:
                ################### FWD pass #####################

                # Decode smpl motion using motion vae
                mvae_model.eval()
                nmg_repr = (smpl2nmg(smpl_motion,smpl).reshape(-1,seq_len,22*9) - mvae_mean)/mvae_std
                smpl_motion_latent = mvae_model.encode(nmg_repr)[:,0]

                # SMPL fwd pass
                smpl_out = smpl.forward(root_orient = smpl_motion[:,3:6],
                                            pose_body = smpl_motion[:,6:],
                                            trans = smpl_motion[:,:3],
                                            betas = smpl_shape.unsqueeze(1).expand(-1,seq_len,-1).reshape(-1,smpl_shape.shape[-1]))

                j3ds = smpl_out.Jtr[:,:22,:]

                # camera intrinsics
                if config["optimize_intr"]:
                    cam_intr =  torch.stack([torch.stack([torch.cat([focal_len[c].unsqueeze(0),torch.tensor([0,im_res[0][0]/2])]),
                            torch.cat([torch.tensor([0]),focal_len[c].unsqueeze(0),torch.tensor([im_res[0][1]/2])]),
                            torch.tensor([0,0,1])]) for c in range(num_cams)]).unsqueeze(0)

                # camera extrinsics
                cam_orient_expanded = torch.cat([x.repeat(1,1,seq_len,1) if x.shape[2]==1 else x for x in cam_orient],dim=1)
                cam_position_expanded = torch.cat([x.repeat(1,1,seq_len,1) if x.shape[2]==1 else x for x in cam_position],dim=1)
                cam_ext_temp = torch.cat([p3d_rt.rotation_6d_to_matrix(cam_orient_expanded),cam_position_expanded.unsqueeze(4)],dim=4)
                cam_ext = torch.cat([cam_ext_temp,torch.tensor([0,0,0,1]).type_as(cam_ext_temp).repeat(batch_size,num_cams,seq_len,1,1)],dim=3)

                # camera extrinsics
                cam_poses = torch.inverse(cam_ext)

                # camera projection
                proj_j3ds = torch.stack([perspective_projection(j3ds,
                                cam_ext[:,i,:,:3,:3].reshape(-1,3,3),
                                cam_ext[:,i,:,:3,3].reshape(-1,3),
                                cam_intr[:,i].unsqueeze(1).expand(-1,seq_len,-1,-1).reshape(-1,3,3)).reshape(batch_size,seq_len,-1,2) for i in range(num_cams)]).permute(1,0,2,3,4)


                ####################### Losses #######################

                # reprojection loss
                loss_2d = (j2ds[:,:,:,:22,2]*(((proj_j3ds[:,:,:,:22] - j2ds[:,:,:,:22,:2])**2).sum(dim=4))).sum(dim=1).mean()

                # latent space regularization
                loss_z = (smpl_motion_latent*smpl_motion_latent).mean()

                # smooth camera motions
                loss_cams = ((cam_poses[:,:,1:] - cam_poses[:,:,:-1])**2).mean()

                # shape regularization loss
                loss_betas = (smpl_shape*smpl_shape).mean()

                # ground penetration loss
                loss_human_gp = torch.nn.functional.relu(-j3ds[:,:,2]).mean()

                # camera ground penetration loss
                loss_cam_gp = torch.nn.functional.relu(-cam_poses[:,:,:,2,3]).mean()

                # loss vposer
                loss_vp = (vp_model.encode(smpl_motion[:,6:]).mean**2).mean()

                # loss smpl in front
                loss_smpl_in_front = torch.nn.functional.relu(-cam_ext[:,:,:,2,3]).mean()

                # total loss
                loss = loss_2d_weight * loss_2d + \
                        loss_z_weight * loss_z + \
                            loss_cams_weight * loss_cams + \
                                loss_betas_weight * loss_betas + \
                                    loss_human_gp_weight * loss_human_gp + \
                                        loss_cam_gp_weight * loss_cam_gp + \
                                            loss_vp_weight * loss_vp + \
                                                loss_smpl_in_front_weight * loss_smpl_in_front
                # zero grad optimizer
                optim.zero_grad()
                
                # backward
                loss.backward()

                # optim step
                optim.step()


                # print loss
                t.set_postfix({"loss":loss.item(),
                                "loss_2d":loss_2d.item(),
                                "loss_z":loss_z.item(),
                                "loss_cams":loss_cams.item(),
                                "loss_betas":loss_betas.item(),
                                "loss_human_gp":loss_human_gp.item(),
                                "loss_cam_gp":loss_cam_gp.item(),
                                "loss_vp":loss_vp.item(),
                                "loss_smpl_front":loss_smpl_in_front.item()})


                # Viz
                if i == n_optim_iters-1:
                # if i%100 == 0:
                    full_images_paths = batch["full_im_paths"]
                    num_cams = len(full_images_paths)
                    seq_len = len(full_images_paths[0])
                    # random index
                    idx = np.random.randint(cam_ext.shape[0])
                    rend_ims = [np.zeros([seq_len,im_res[c][1],im_res[c][0],3]) for c in range(num_cams)]
                    for cam in tqdm(range(num_cams)):
                        for s in range(seq_len):
                            im = cv2.imread(full_images_paths[cam][s])[:im_res[cam][1],:im_res[cam][0],::-1]/255.
                            rend_ims[cam][s] = renderer[cam](smpl_out.v.view(-1,seq_len,6890,3)[idx,s].detach().cpu().numpy(),
                                                cam_ext[idx,cam,s,:3,3].detach().cpu().numpy(),
                                                cam_ext[idx,cam,s,:3,:3].unsqueeze(0).detach().cpu().numpy(),
                                                im,intr=cam_intr[idx,cam].detach().cpu().numpy(),
                                                faces=smpl.f.detach().cpu().numpy())
                    for c in range(num_cams):
                        imageio.mimsave("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/test_{:04d}_{:05d}_cam_{:02d}.gif".format(config["trial_name"],
                                    seq_no,i,seq_start,c),list(rend_ims[c][:,::5,::5]))
                        # imageio.mimsave("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/test_{:05d}_cam_{:02d}.gif".format(config["trial_name"],seq_no,seq_start,c),
                        #             [make_grid(torch.from_numpy(rend_ims[:,i]).permute(0,3,1,2),
                        #                 nrow=rend_ims.shape[0]).permute(1,2,0).cpu().numpy()[::5,::5] for i in range(rend_ims.shape[1])])

        with torch.no_grad():
            # full_cam_orient = p3d_rt.matrix_to_quaternion(cam_poses[:,:,:,:3,:3]).detach().cpu().numpy()
            # full_cam_position = cam_poses[:,:,:,:3,3].detach().cpu().numpy()
            # full_smpl_verts = smpl_out.v.detach().cpu().numpy()
            # full_smpl_shape = smpl_shape.detach().cpu().numpy()

            np.savez("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/test_{:05d}".format(config["trial_name"],seq_no,seq_start),
                verts=smpl_out.v.detach().cpu().numpy(),
                cam_trans=cam_poses[:,:,:,:3,3].detach().cpu().numpy(),
                cam_rots=p3d_rt.matrix_to_quaternion(cam_poses[:,:,:,:3,:3]).detach().cpu().numpy())
            
            if len(full_cam_orient) == 0:
                full_cam_orient.append(p3d_rt.matrix_to_quaternion(cam_poses[:,:,:,:3,:3]).detach().cpu().numpy())
                full_cam_position.append(cam_poses[:,:,:,:3,3].detach().cpu().numpy())
                full_smpl_verts.append(smpl_out.v.detach().cpu().numpy())
                full_smpl_shape.append(smpl_shape.detach().cpu().numpy())
                full_smpl_motion_latent.append(smpl_motion_latent.detach())
                full_cam_ext.append(cam_ext.detach())
                # save last smpl orient
                last_smpl_orient = smpl_motion[:,3:6].detach()
                last_smpl_trans = smpl_motion[:,:3].detach()
            else:
                pos, ori = geometry.get_ground_point(last_smpl_trans[-overlap+1],p3d_rt.axis_angle_to_matrix(last_smpl_orient[-overlap+1]))
                prev_tfm = to_homogeneous(ori,pos)
                curr_pos, curr_ori = geometry.get_ground_point(smpl_motion[0,:3],p3d_rt.axis_angle_to_matrix(smpl_motion[0,3:6]))
                curr_tfm = to_homogeneous(curr_ori,curr_pos)
                tfm = torch.matmul(prev_tfm,torch.inverse(curr_tfm))
                curr_smpl = to_homogeneous(p3d_rt.axis_angle_to_matrix(smpl_motion[:,3:6]),smpl_motion[:,:3])
                tfmd_smpl = torch.matmul(tfm,curr_smpl)
                updated_smpl_orient = p3d_rt.matrix_to_axis_angle(tfmd_smpl[:,:3,:3])
                updated_smpl_trans = tfmd_smpl[:,:3,3]

                smpl_out_updated = smpl.forward(root_orient = updated_smpl_orient,
                                            pose_body = smpl_motion[:,6:],
                                            trans = updated_smpl_trans,
                                            betas = smpl_shape.unsqueeze(1).expand(-1,seq_len,-1).reshape(-1,smpl_shape.shape[-1]))
                updated_cam_orient = torch.matmul(tfm[:,:3,:3],cam_poses[0,:,:,:3,:3])
                updated_cam_position = torch.matmul(tfm[:,:3,:3],cam_poses[0,:,:,:3,3:]).squeeze(3) + tfm[0,:3,3]
                full_cam_orient.append(p3d_rt.matrix_to_quaternion(updated_cam_orient).unsqueeze(0).detach().cpu().numpy())
                full_cam_position.append(updated_cam_position.unsqueeze(0).detach().cpu().numpy())
                full_smpl_verts.append(smpl_out_updated.v.detach().cpu().numpy())
                full_smpl_shape.append(smpl_shape.detach().cpu().numpy())
                full_smpl_motion_latent.append(smpl_motion_latent.detach())
                full_cam_ext.append(cam_ext.detach())

                # save last smpl orient
                last_smpl_orient = p3d_rt.matrix_to_axis_angle(tfmd_smpl[:,:3,:3]).detach()
                last_smpl_trans = tfmd_smpl[:,:3,3].detach()


full_cam_orient = np.concatenate(full_cam_orient,axis=2)
full_cam_position = np.concatenate(full_cam_position,axis=2)
full_smpl_verts = np.concatenate(full_smpl_verts)
full_smpl_shape = np.concatenate(full_smpl_shape)
full_smpl_motion_latent = torch.cat(full_smpl_motion_latent,dim=0)
full_cam_ext = torch.cat(full_cam_ext,dim=0)
# imageio.mimsave("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/test/test.gif",overlay_gifs,fps=30)

np.savez("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/test/{:04d}/test_full".format(seq_no),
                verts=full_smpl_verts,
                cam_trans=full_cam_position,
                cam_rots=full_cam_orient)


non_overlap_cam_position = np.concatenate([full_cam_position[:,:,i:i+seq_len-overlap] for i in range(0,full_cam_position.shape[2]-seq_len,seq_len)] + 
                            [full_cam_position[:,:,-seq_len:]],axis=2)
non_overlap_cam_orient = np.concatenate([full_cam_orient[:,:,i:i+seq_len-overlap] for i in range(0,full_cam_orient.shape[2]-seq_len,seq_len)] +
                            [full_cam_orient[:,:,-seq_len:]],axis=2)
non_overlap_smpl_verts = np.concatenate([full_smpl_verts[i:i+seq_len-overlap] for i in range(0,full_smpl_verts.shape[0]-seq_len,seq_len)] +
                            [full_smpl_verts[-seq_len:]],axis=0)

np.savez("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/test/{:04d}/test_full_non_overlap".format(seq_no),
                verts=non_overlap_smpl_verts,
                cam_trans=non_overlap_cam_position,
                cam_rots=non_overlap_cam_orient)




###############################################################################################################################


# SLERP interpolation
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
full_cam_interp_orient = []
full_cam_interp_position = []
for i in range(num_cams):
    rots = p3d_rt.quaternion_to_matrix(torch.from_numpy(non_overlap_cam_orient[0,i]))
    idcs = list(range(0,rots.shape[0]-seq_len+1,seq_len-overlap)) + [rots.shape[0]-1]
    key_rots = R.from_matrix(rots[idcs])
    interp_rots = torch.from_numpy(Slerp(idcs,key_rots)(range(0,rots.shape[0])).as_matrix())
    full_cam_interp_orient.append(p3d_rt.matrix_to_quaternion(interp_rots).detach().cpu().numpy())
    full_cam_interp_position.append(interp1d(idcs,non_overlap_cam_position[0,i][idcs],axis=0)(range(0,rots.shape[0])))

full_cam_interp_orient = np.stack(full_cam_interp_orient,axis=0)
full_cam_interp_position = np.stack(full_cam_interp_position,axis=0)


np.savez("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/test/{:04d}/test_full_slerp".format(seq_no),
     verts=non_overlap_smpl_verts,
     cam_trans=full_cam_interp_position[np.newaxis, :],
     cam_rots=full_cam_interp_orient[np.newaxis, :])


#################################################################################################################


loss_2d_weight = 1
loss_z_weight = 100
loss_cams_weight = 10000
loss_betas_weight = 10
loss_cont_weight = 100


# transformations of cameras
tfmd_full_cam_ext = torch.cat([p3d_rt.quaternion_to_matrix(torch.from_numpy(full_cam_interp_orient)),torch.from_numpy(full_cam_interp_position).unsqueeze(3)],dim=3)
tfmd_full_cam_ext = torch.cat([tfmd_full_cam_ext,torch.tensor([0,0,0,1]).type_as(tfmd_full_cam_ext).repeat(tfmd_full_cam_ext.shape[0]
                                    ,tfmd_full_cam_ext.shape[1],1,1)],dim=2)
tfmd_full_cam_ext = torch.inverse(tfmd_full_cam_ext)
tfmd_full_cam_ext = torch.cat([tfmd_full_cam_ext[:,i:i+seq_len] for i in range(0,tfmd_full_cam_ext.shape[1]-seq_len+1,seq_len-overlap)],dim=1)


# Camera and SMPL params
cam_orient_staticcam = p3d_rt.matrix_to_rotation_6d(tfmd_full_cam_ext[:,:,:3,:3]).detach().float().requires_grad_(True)
cam_position_staticcam = tfmd_full_cam_ext[:,:,:3,3].detach().float().requires_grad_(True)
# cam_orient_movingcam = p3d_rt.matrix_to_rotation_6d(p3d_rt.axis_angle_to_matrix(torch.tensor([3.14/2,0,0]).float())).repeat(batch_size,num_cams-1,seq_len,1).requires_grad_(True)
# cam_position_movingcam = torch.tensor([0,0,5]).float().repeat(batch_size,num_cams-1,seq_len,1).requires_grad_(True)
smpl_shape = torch.zeros(10).unsqueeze(0).requires_grad_(True)


j2ds = []

# with trange(451,ds.data_lengths[seq_no]-2000,2*(seq_len-overlap)) as seq_t:
with trange(big_seq_start,big_seq_end,2*(seq_len-overlap)) as seq_t:
    for seq_start in seq_t:
        # get batch
        batch = ds.__getitem__(seq_no,seq_start)
        j2ds.append(batch["j2d"].float().unsqueeze(0))
        cam_intr = torch.from_numpy(batch["cam_intr"]).float().unsqueeze(0)

j2ds = torch.cat(j2ds,dim=0)
smpl_motion_latent = full_smpl_motion_latent.detach().requires_grad_(True)

################# Optimizer #################
optim = torch.optim.Adam([smpl_motion_latent, smpl_shape],lr=lr)

batch_size = j2ds.shape[0]

with trange(n_optim_iters) as t:
    for _ in t:
        ################### FWD pass #####################

        # Decode smpl motion using motion vae
        mvae_model.eval()
        smpl_motion_decoded = mvae_model.decode(smpl_motion_latent)
        smpl_motion_unnorm = smpl_motion_decoded*mvae_std + mvae_mean
        smpl_motion = nmg2smpl(smpl_motion_unnorm.reshape(batch_size*seq_len,22,9),smpl).reshape(batch_size,seq_len,-1)

        # transform seq chunks
        cont_seq = [smpl_motion[0].clone()]
        for j in range(1,smpl_motion.shape[0]):
            pos, ori = geometry.get_ground_point(smpl_motion[j-1,-overlap,:3],p3d_rt.axis_angle_to_matrix(smpl_motion[j-1,-overlap,3:6]))
            seq_temp_ori = p3d_rt.matrix_to_axis_angle(torch.matmul(ori,p3d_rt.axis_angle_to_matrix(smpl_motion[j,:,3:6])))
            seq_temp_pos = torch.matmul(ori,smpl_motion[j,:,:3].unsqueeze(2)).squeeze(2) + pos
            cont_seq.append(torch.cat([seq_temp_pos,seq_temp_ori,smpl_motion[j,:,6:]],dim=1))

        cont_seq = torch.cat(cont_seq)

        # SMPL fwd pass
        cont_seq = cont_seq.reshape(-1,cont_seq.shape[-1])
        smpl_out = smpl.forward(root_orient = cont_seq[:,3:6],
                                    pose_body = cont_seq[:,6:],
                                    trans = cont_seq[:,:3],
                                    betas = smpl_shape.unsqueeze(1).expand(-1,cont_seq.shape[0],-1).reshape(-1,smpl_shape.shape[-1]))

        j3ds = smpl_out.Jtr[:,:22,:]

        # camera extrinsics
        # cam_orient = torch.cat([cam_orient_staticcam.repeat(1,1,seq_len,1),cam_orient_movingcam],dim=1)
        # cam_position = torch.cat([cam_position_staticcam.repeat(1,1,seq_len,1),cam_position_movingcam],dim=1)
        cam_orient_seq = cam_orient_staticcam
        cam_position_seq = cam_position_staticcam
        cam_ext = torch.cat([p3d_rt.rotation_6d_to_matrix(cam_orient_seq),cam_position_seq.unsqueeze(3)],dim=3).permute(1,0,2,3)
        cam_ext = torch.cat([cam_ext,torch.tensor([0,0,0,1]).type_as(cam_ext).repeat(j3ds.shape[0],num_cams,1,1)],dim=2)

        # camera projection
        proj_j3ds = torch.stack([perspective_projection(j3ds,
                        cam_ext[:,i,:3,:3].reshape(-1,3,3),
                        cam_ext[:,i,:3,3].reshape(-1,3),
                        cam_intr[:,i].expand(j3ds.shape[0],-1,-1).float()).reshape(batch_size,seq_len,-1,2) for i in range(num_cams)]).permute(1,0,2,3,4)

        # camera extrinsics
        cam_poses = torch.inverse(cam_ext)

        ####################### Losses #######################

        # reprojection loss
        loss_2d = (j2ds[:,:,:,:22,2]*(((proj_j3ds[:,:,:,:22] - j2ds[:,:,:,:22,:2])**2).sum(dim=4))).mean()

        # latent space regularization
        loss_z = (smpl_motion_latent*smpl_motion_latent).mean()

        # smooth camera motions
        loss_cams = ((cam_poses[1:] - cam_poses[:-1])**2).mean()

        # shape regularization loss
        loss_betas = (smpl_shape*smpl_shape).mean()

        # continuation loss
        cont_seq_reshaped = cont_seq.reshape(smpl_motion.shape[0],smpl_motion.shape[1],cont_seq.shape[-1])
        loss_cont = torch.cat([((cont_seq_reshaped[i,-overlap:]-cont_seq_reshaped[i+1,:overlap])**2).mean().unsqueeze(0) for i in range(cont_seq_reshaped.shape[0]-1)]).mean()

        # total loss
        loss = loss_2d_weight * loss_2d + \
                loss_z_weight * loss_z + \
                    loss_cams_weight * loss_cams + \
                        loss_betas_weight * loss_betas + \
                            loss_cont_weight * loss_cont
        # zero grad optimizer
        optim.zero_grad()
        
        # backward
        loss.backward()

        # optim step
        optim.step()


        # print loss
        t.set_postfix({"loss":loss.item(),
                        "loss_2d":loss_2d.item(),
                        "loss_z":loss_z.item(),
                        "loss_cams":loss_cams.item(),
                        "loss_betas":loss_betas.item(),
                        "loss_cont":loss_cont.item()})

with torch.no_grad():
    full_cam_orient = p3d_rt.matrix_to_quaternion(cam_poses[:,:,:3,:3]).detach().cpu().numpy()
    full_cam_orient = full_cam_orient.reshape(full_smpl_motion_latent.shape[0],seq_len,num_cams,4)
    full_cam_orient = np.concatenate([full_cam_orient[0]]+[full_cam_orient[i,overlap:] for i in range(1,full_cam_orient.shape[0])],axis=0)
    full_cam_position = cam_poses[:,:,:3,3].detach().cpu().numpy()
    full_cam_position = full_cam_position.reshape(full_smpl_motion_latent.shape[0],seq_len,num_cams,3)
    full_cam_position = np.concatenate([full_cam_position[0]]+[full_cam_position[i,overlap:] for i in range(1,full_cam_position.shape[0])],axis=0)
    full_smpl_verts = smpl_out.v.detach().cpu().numpy()
    full_smpl_verts = full_smpl_verts.reshape(full_smpl_motion_latent.shape[0],seq_len,smpl_out.v.shape[1],3)
    full_smpl_verts = np.concatenate([full_smpl_verts[0]]+[full_smpl_verts[i,overlap:] for i in range(1,full_smpl_verts.shape[0])],axis=0)
    full_smpl_shape = smpl_shape.detach().cpu().numpy()

np.savez("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/test/{:04d}/test_{:05d}_stage2".format(seq_no,seq_start),
    verts=full_smpl_verts,
    cam_trans=full_cam_position.transpose(1, 0, 2)[np.newaxis],
    cam_rots=full_cam_orient.transpose(1, 0, 2)[np.newaxis])
            