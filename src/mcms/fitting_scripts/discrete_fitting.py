import os
os.environ["PYOPENGL_PLATFORM"] = 'egl'
from mcms.models import mcms
import yaml
import torch
from pytorch3d.transforms import rotation_conversions as p3d_rt
import imageio
from torchvision.utils import make_grid

ckpt_path = "/is/ps3/nsaini/projects/mcms/mcms_logs/2022_03_07/v000_/checkpoints/epoch=2099-step=125999.ckpt"
hparams = yaml.safe_load(open("/".join(ckpt_path.split("/")[:-2])+"/hparams.yaml","r"))

from mcms.dsets import h36m, copenet_real
from torch.utils.data import DataLoader
from nmg.models import nmg
import numpy as np
from tqdm import tqdm, trange
from mcms.utils.utils import nmg2smpl
from human_body_prior.body_model.body_model import BodyModel
from mcms.utils.geometry import perspective_projection
from mcms.utils.renderer import Renderer
import cv2
from mcms.utils import geometry
from mcms.utils.utils import to_homogeneous


# Params
batch_size = 1
seq_len = 25
loss_2d_weight = 1
loss_z_weight = 10
loss_cams_weight = 100
loss_betas_weight = 10
n_optim_iters = 1000
loss_j3d_weight = 1000
lr = 0.01

# dataset
dset = "copenet_real"

# get seq number
seq_no = 29
big_seq_start = 0
big_seq_end = 50
overlap = 20


# SMPL
smpl = BodyModel(bm_fname=hparams["model_smpl_neutral_path"])

# Motion VAE
nmg_hparams = yaml.safe_load(open("/".join(hparams["train_motion_vae_ckpt_path"].split("/")[:-2])+"/hparams.yaml"))
mvae_model = nmg.nmg.load_from_checkpoint(hparams["train_motion_vae_ckpt_path"],nmg_hparams)
mean_std = np.load(hparams["model_mvae_mean_std_path"])
mvae_mean = torch.from_numpy(mean_std["mean"]).float()
mvae_std = torch.from_numpy(mean_std["std"]).float()

# Dataloader
if dset.lower() == "h36m":
    ds = h36m.h36m(hparams)
    fps_scl = 2
    # renderer
    im_res = [1000,1000]
elif dset.lower() == "copenet_real":
    hparams["data_datapath"] = "/home/nsaini/Datasets/copenet_data"
    ds = copenet_real.copenet_real(hparams,range(0,7000))
    fps_scl = 1
    im_res=[1920,1080]
renderer = Renderer(img_res=im_res)
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
os.makedirs("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/test/{:04d}".format(seq_no),exist_ok=True)

# with trange(451,ds.data_lengths[seq_no]-50,50) as seq_t:
with trange(big_seq_start,big_seq_end,fps_scl*(seq_len-overlap)) as seq_t:
    for seq_start in seq_t:
        if dset.lower() == "h36m":
            # get batch
            batch = ds.__getitem__(seq_no,seq_start)
        else:
            batch = ds.__getitem__(seq_start)
        j2ds = batch["j2d"].float().unsqueeze(0)
        cam_intr = torch.from_numpy(batch["cam_intr"]).float().unsqueeze(0)
        batch_size = j2ds.shape[0]
        num_cams = j2ds.shape[1]
        seq_len = j2ds.shape[2]

        # Camera and SMPL params
        # cam_orient_staticcam = p3d_rt.matrix_to_rotation_6d(p3d_rt.axis_angle_to_matrix(torch.tensor([3.14/2,0,0]).float())).repeat(batch_size,num_cams,1,1).requires_grad_(True)
        # cam_position_staticcam = torch.tensor([0,0,5]).float().repeat(batch_size,num_cams,1,1).requires_grad_(True)
        cam_orient_movingcam = p3d_rt.matrix_to_rotation_6d(p3d_rt.axis_angle_to_matrix(torch.tensor([3.14/2,0,0]).float())).repeat(batch_size,num_cams,seq_len,1).requires_grad_(True)
        cam_position_movingcam = torch.tensor([0,0,10]).float().repeat(batch_size,num_cams,seq_len,1).requires_grad_(True)
        smpl_motion_latent = torch.zeros(1024).unsqueeze(0).requires_grad_(True)
        smpl_shape = torch.zeros(10).unsqueeze(0).requires_grad_(True)
        

        ################# Optimizer #################
        optim = torch.optim.Adam([cam_orient_movingcam,cam_position_movingcam,smpl_motion_latent, smpl_shape],lr=lr)

        with trange(n_optim_iters) as t:
            for i in t:
                ################### FWD pass #####################

                # Decode smpl motion using motion vae
                mvae_model.eval()
                smpl_motion_decoded = mvae_model.decode(smpl_motion_latent)
                smpl_motion_unnorm = smpl_motion_decoded*mvae_std + mvae_mean
                smpl_motion = nmg2smpl(smpl_motion_unnorm.reshape(batch_size*seq_len,22,9),smpl).reshape(batch_size,seq_len,-1)

                # SMPL fwd pass
                smpl_motion = smpl_motion.reshape(-1,smpl_motion.shape[-1])
                smpl_out = smpl.forward(root_orient = smpl_motion[:,3:6],
                                            pose_body = smpl_motion[:,6:],
                                            trans = smpl_motion[:,:3],
                                            betas = smpl_shape.unsqueeze(1).expand(-1,seq_len,-1).reshape(-1,smpl_shape.shape[-1]))

                j3ds = smpl_out.Jtr[:,:22,:]

                # camera extrinsics
                # cam_orient_seq = cam_orient_staticcam.repeat(1,1,seq_len,1)
                cam_orient_seq = cam_orient_movingcam
                # cam_position_seq = cam_position_staticcam.repeat(1,1,seq_len,1)
                cam_position_seq = cam_position_movingcam
                cam_ext = torch.cat([p3d_rt.rotation_6d_to_matrix(cam_orient_seq),cam_position_seq.unsqueeze(4)],dim=4)
                cam_ext = torch.cat([cam_ext,torch.tensor([0,0,0,1]).type_as(cam_ext).repeat(batch_size,num_cams,seq_len,1,1)],dim=3)

                # camera projection
                proj_j3ds = torch.stack([perspective_projection(j3ds,
                                cam_ext[:,i,:,:3,:3].reshape(-1,3,3),
                                cam_ext[:,i,:,:3,3].reshape(-1,3),
                                cam_intr[:,i].unsqueeze(1).expand(-1,seq_len,-1,-1).reshape(-1,3,3).float()).reshape(batch_size,seq_len,-1,2) for i in range(num_cams)]).permute(1,0,2,3,4)

                # camera extrinsics
                cam_poses = torch.inverse(cam_ext)

                ####################### Losses #######################

                # reprojection loss
                loss_2d = (j2ds[:,:,:,:22,2]*(((proj_j3ds[:,:,:,:22] - j2ds[:,:,:,:22,:2])**2).sum(dim=4))).mean()

                # latent space regularization
                loss_z = (smpl_motion_latent*smpl_motion_latent).mean()

                # smooth camera motions
                loss_cams = ((cam_poses[:,:,1:] - cam_poses[:,:,:-1])**2).mean()

                # shape regularization loss
                loss_betas = (smpl_shape*smpl_shape).mean()

                # total loss
                loss = loss_2d_weight * loss_2d + \
                        loss_z_weight * loss_z + \
                            loss_cams_weight * loss_cams + \
                                loss_betas_weight * loss_betas
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
                                "loss_betas":loss_betas.item()})


                # Viz
                if i == n_optim_iters-1:
                    full_images_paths = batch["full_im_paths"]
                    num_cams = len(full_images_paths)
                    seq_len = len(full_images_paths[0])
                    if hparams["data_name"].lower() == "h36m":
                        im_res = [1000,1000]

                    # random index
                    idx = np.random.randint(cam_orient_seq.shape[0])
                    rend_ims = np.zeros([num_cams,seq_len,im_res[1],im_res[0],3])
                    for cam in range(num_cams):
                        for s in range(seq_len):
                            im = cv2.imread(full_images_paths[cam][s])[:im_res[1],:im_res[0],::-1]/255.
                            rend_ims[cam,s] = renderer(smpl_out.v.view(-1,seq_len,6890,3)[idx,s].detach().cpu().numpy(),
                                                cam_position_seq[idx,cam,s,:3].detach().cpu().numpy(),
                                                p3d_rt.rotation_6d_to_matrix(cam_orient_seq[idx,cam,s].unsqueeze(0)).detach().cpu().numpy(),
                                                im,intr=cam_intr[idx,cam].detach().cpu().numpy(),
                                                faces=smpl.f.detach().cpu().numpy())

                    imageio.mimsave("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/test/{:04d}/test_{:05d}.gif".format(seq_no,seq_start),
                                    [make_grid(torch.from_numpy(rend_ims[:,i]).permute(0,3,1,2),
                                        nrow=rend_ims.shape[0]).permute(1,2,0).cpu().numpy()[::5,::5] for i in range(rend_ims.shape[1])])

        with torch.no_grad():
            # full_cam_orient = p3d_rt.matrix_to_quaternion(cam_poses[:,:,:,:3,:3]).detach().cpu().numpy()
            # full_cam_position = cam_poses[:,:,:,:3,3].detach().cpu().numpy()
            # full_smpl_verts = smpl_out.v.detach().cpu().numpy()
            # full_smpl_shape = smpl_shape.detach().cpu().numpy()

            np.savez("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/test/{:04d}/test_{:05d}".format(seq_no,seq_start),
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

np.savez("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/test/test",
                verts=full_smpl_verts,
                cam_trans=full_cam_position,
                cam_rots=full_cam_orient)




###############################################################################################################################

loss_2d_weight = 1
loss_z_weight = 100
loss_cams_weight = 10000
loss_betas_weight = 10
loss_cont_weight = 100
loss_keyframes_weight = 1000

# transformations of cameras
tfmd_full_cam_ext = torch.cat([p3d_rt.quaternion_to_matrix(torch.from_numpy(full_cam_orient)),torch.from_numpy(full_cam_position).unsqueeze(3)],dim=3)
tfmd_full_cam_ext = torch.cat([tfmd_full_cam_ext,torch.tensor([0,0,0,1]).type_as(tfmd_full_cam_ext).repeat(tfmd_full_cam_ext.shape[0]
                                    ,tfmd_full_cam_ext.shape[1],1,1)],dim=2)
tfmd_full_cam_ext = torch.inverse(tfmd_full_cam_ext)

# Camera and SMPL params
cam_orient_keyframes = p3d_rt.matrix_to_rotation_6d(tfmd_full_cam_ext[seq_len//2::seq_len-overlap,:,:3,:3]).permute(1,0,2).detach()
cam_position_keyframes = tfmd_full_cam_ext[seq_len//2::seq_len-overlap,:,:3,3].permute(1,0,2).detach()
cam_orient_movingcam = p3d_rt.matrix_to_rotation_6d(tfmd_full_cam_ext[:,:,:3,:3]).permute(1,0,2).detach().requires_grad_(True)
cam_position_movingcam = tfmd_full_cam_ext[:,:,:3,3].permute(1,0,2).detach().requires_grad_(True)
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
optim = torch.optim.Adam([cam_orient_movingcam,cam_position_movingcam,smpl_motion_latent, smpl_shape],lr=lr)

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

        cont_seq = torch.stack(cont_seq)

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
        cam_orient_seq = torch.cat([cam_orient_movingcam[:,i:i+seq_len] for i in range(0,cam_orient_movingcam.shape[1]-seq_len+overlap,seq_len-overlap)],dim=1).permute(1,0,2)
        cam_position_seq = torch.cat([cam_position_movingcam[:,i:i+seq_len] for i in range(0,cam_orient_movingcam.shape[1]-seq_len+overlap,seq_len-overlap)],dim=1).permute(1,0,2)
        cam_ext = torch.cat([p3d_rt.rotation_6d_to_matrix(cam_orient_seq),cam_position_seq.unsqueeze(3)],dim=3)
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

        # loss keyframes
        loss_kf = ((cam_orient_movingcam[:,(seq_len//2)::seq_len-overlap] - cam_orient_keyframes)**2).mean() + \
                    ((cam_position_movingcam[:,(seq_len//2)::seq_len-overlap] - cam_position_keyframes)**2).mean()

        # continuation loss
        cont_seq_reshaped = cont_seq.reshape(smpl_motion.shape[0],smpl_motion.shape[1],cont_seq.shape[-1])
        loss_cont = torch.cat([((cont_seq_reshaped[i,-overlap:]-cont_seq_reshaped[i+1,:overlap])**2).mean().unsqueeze(0) for i in range(cont_seq_reshaped.shape[0]-1)]).mean()

        # total loss
        loss = loss_2d_weight * loss_2d + \
                loss_z_weight * loss_z + \
                    loss_cams_weight * loss_cams + \
                        loss_betas_weight * loss_betas + \
                            loss_cont_weight * loss_cont + \
                                loss_keyframes_weight * loss_kf
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
                        "loss_cont":loss_cont.item(),
                        "loss_kf":loss_kf.item()})

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

np.savez("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/test/{:04d}/test_{:05d}".format(seq_no,seq_start),
    verts=full_smpl_verts,
    cam_trans=full_cam_position.transpose(1, 0, 2)[np.newaxis],
    cam_rots=full_cam_orient.transpose(1, 0, 2)[np.newaxis])
            