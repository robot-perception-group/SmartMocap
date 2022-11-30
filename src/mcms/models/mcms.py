from pyexpat import model
import pytorch_lightning as pl
import torch
import cv2
import torchvision
import torchvision.models.resnet as resnet
import torch.nn as nn
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder
from torch.nn import functional as F
from torch.nn.modules import LayerNorm
import numpy as np
from pytorch3d.transforms import rotation_conversions as p3d_rt
from ..utils.geometry import batch_rodrigues, perspective_projection, estimate_translation, rot6d_to_rotmat
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from ..utils.utils import mop2smpl, smpl2mop 
from ..utils.geometry import perspective_projection
from ..utils.renderer import Renderer
from . import backbone
from ..dsets import h36m
from mop.models import mop
import yaml
import imageio
from koila import lazy
import os
from os.path import join as ospj
from torchvision.utils import make_grid

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser


class mcms(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # define resnet backbone
        self.backbone = backbone.getbackbone()

        if hparams["model_zspace"].lower() == "vposer":
            npose = 32          
            dummy_zeros = 2         # two dummy zeros for transformer layers
        elif hparams["model_zspace"].lower() == "motion_vae":
            npose = 1024
            dummy_zeros = 2
        elif hparams["model_zspace"].lower() == "smpl":
            npose = 21*6    
            dummy_zeros = 4         # four dummy zeros for transformer layers
        else:
            import ipdb;ipdb.set_trace()
            
        # transformer encoder layers
        in_dim = 2048 + 3 + 3 + 6 + npose + 10 + dummy_zeros
        dropout = 0.1
        enc_layer = TransformerEncoderLayer(in_dim, nhead=8, dim_feedforward=1024, dropout=dropout)
        self.interm_encoder_layers = TransformerEncoder(enc_layer, hparams["model_num_enc_layers"], LayerNorm(in_dim))
        if self.hparams["model_zspace"].lower() == "motion_vae":
            self.deccam = nn.Linear(in_dim, 3 + 6)
            self.decpose = nn.Linear(in_dim, npose)
            self.decshape = nn.Linear(in_dim, 10)
            self.decpose_pool = nn.Linear(npose*hparams["model_num_cams"]*hparams["data_seq_len"], npose)
            self.decshape_pool = nn.Linear(10*hparams["model_num_cams"]*hparams["data_seq_len"], 10)
            nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decpose_pool.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decshape_pool.weight, gain=0.01)
        else:
            self.decpose = nn.Linear(in_dim, 3 + 6 + npose)
            self.decshape = nn.Linear(in_dim, 10)
            nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)

        # init params
        mean_params = np.load("/is/ps3/nsaini/projects/trcopenet/src/trcopenet/data/smpl_mean_params.npz")
        if hparams["model_zspace"].lower() == "motion_vae":
            init_orient = p3d_rt.matrix_to_rotation_6d(p3d_rt.axis_angle_to_matrix(torch.tensor([90,0,0]).unsqueeze(0).float()))
            init_pose = torch.zeros(1024).unsqueeze(0)
            init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
            init_position = torch.tensor([0,0,5]).float().unsqueeze(0)
        elif hparams["model_zspace"].lower() == "vposer":
            init_orient = torch.tensor([1,0,0,0,1,0]).float().unsqueeze(0)
            init_pose = torch.zeros(32).unsqueeze(0)
            init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
            init_position = torch.tensor([0,0,5]).float().unsqueeze(0)
        else:
            init_orient = torch.from_numpy(mean_params['pose'][:6]).unsqueeze(0)
            init_pose = torch.from_numpy(mean_params['pose'][6:]).unsqueeze(0)
            init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
            init_position = torch.tensor([0,0,5]).float().unsqueeze(0)
        self.register_buffer('init_orient', init_orient)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_position', init_position)
        self.register_buffer('init_shape', init_shape)

        self.register_buffer("j_regr",torch.from_numpy(np.load(hparams["train_joint_regressor"])).float().unsqueeze(0))

        # vposer and smpl
        self.vp_model = load_model(hparams["model_vposer_path"], model_code=VPoser,remove_words_in_model_weights="vp_model.")[0]
        self.vp_model.eval()
        for p in self.vp_model.parameters():
            p.requires_grad = False
        if hparams["model_zspace"].lower() == "motion_vae":
            mop_hparams = yaml.safe_load(open("/".join(hparams["train_motion_vae_ckpt_path"].split("/")[:-2])+"/hparams.yaml"))
            self.mvae_model = mop.mop.load_from_checkpoint(hparams["train_motion_vae_ckpt_path"],mop_hparams)
            mean_std = np.load(self.hparams["model_mvae_mean_std_path"])
            self.register_buffer("mvae_mean", torch.from_numpy(mean_std["mean"]).float())
            self.register_buffer("mvae_std", torch.from_numpy(mean_std["std"]).float())
            self.mvae_model.freeze()

        self.smpl = BodyModel(bm_fname=hparams["model_smpl_neutral_path"])
        self.smpl.eval()
        for p in self.smpl.parameters():
            p.requires_grad = False

        
        self.renderer = Renderer(img_res=[1000,1000])

        # # 
        # if hparams["data"].lower() == "agora_unreal_4view":
        #     self.focal_length = [1475,1475] 
        #     self.renderer = Renderer(focal_length=self.focal_length, img_res=[1920,1080], faces=self.smpl.f.detach().cpu().numpy())
        # elif hparams["data"].lower() == "h36m":
        #     import ipdb;ipdb.set_trace()


    def forward(self, x,
                 bb,
                 trans_scale,
                 *args, **kwargs):

        
        batch_size = x.shape[0]
        num_cams = x.shape[1]
        seq_size = x.shape[2]
        
        init_orient = self.init_orient.expand(batch_size,num_cams,seq_size,-1)
        init_theta = self.init_pose.expand(batch_size,num_cams,seq_size,-1)
        init_shape = self.init_shape.expand(batch_size,num_cams,seq_size,-1)
        init_position = self.init_position.expand(batch_size,num_cams,seq_size, -1).clone()
        
        init_position *= trans_scale
        
        # Feed images in the network to predict camera and SMPL parameters 
        xf = self.forward_feat_ext(x.view(-1,3,224,224)).view(batch_size,num_cams,seq_size,-1)

        if self.hparams["model_zspace"].lower() == "vposer":
            dummy_zeros = 2         # two dummy zeros for transformer layers
        elif self.hparams["model_zspace"].lower() == "motion_vae":
            dummy_zeros = 2
        elif self.hparams["model_zspace"].lower() == "smpl":  
            dummy_zeros = 4         # four dummy zeros for transformer layers
        else:
            import ipdb;ipdb.set_trace()
        
        
        if self.hparams["model_zspace"].lower() == "motion_vae":
            if self.hparams["model_use_pe"]:
                pe1,pe2 = torch.meshgrid(torch.linspace(-1,1,num_cams).type_as(xf),torch.linspace(-1,1,seq_size).type_as(xf))
                pe = torch.cat([pe1.unsqueeze(2),pe2.unsqueeze(2)],dim=2)
                tr_in = torch.cat([xf,bb,init_position,init_orient,init_theta,init_shape,pe.unsqueeze(0).expand(batch_size,-1,-1,-1),
                        torch.zeros(batch_size,num_cams,seq_size,dummy_zeros-2).type_as(xf)],dim=3)
            else:
                tr_in = torch.cat([xf,bb,init_position,init_orient,init_theta,init_shape,torch.zeros(batch_size,num_cams,seq_size,dummy_zeros).type_as(xf)],dim=3)
            interm_out = self.interm_encoder_layers(tr_in.reshape(batch_size,num_cams*seq_size,tr_in.shape[-1])).reshape(batch_size,num_cams,seq_size,tr_in.shape[-1])
            # camera poses
            cam_poses = torch.cat([init_position, init_orient],3) + self.deccam(interm_out)
            cam_poses[:,:,:,:3] /= trans_scale
            # betas
            pred_betas_interm = self.decshape(interm_out)
            pred_betas = self.init_shape + self.decshape_pool(pred_betas_interm.reshape(batch_size,num_cams*pred_betas_interm.shape[-2]*pred_betas_interm.shape[-1]))
            # motion Z
            pred_motion_z_interm = self.decpose(interm_out)
            pred_motion_z = self.init_pose + self.decpose_pool(pred_motion_z_interm.reshape(batch_size,num_cams*pred_motion_z_interm.shape[-2]*pred_motion_z_interm.shape[-1]))
            
            # Motion decoder forward
            self.mvae_model.eval()
            pred_pose = self.mvae_model.decode(pred_motion_z)
            # Normalization
            pred_pose_unnorm = pred_pose*self.mvae_std + self.mvae_mean
            # mop representation to smpl
            pred_pose_smpl = mop2smpl(pred_pose_unnorm.reshape(batch_size*seq_size,22,9),self.smpl).reshape(batch_size,seq_size,-1)
            
        return cam_poses, pred_pose_smpl, pred_betas, pred_motion_z


    def forward_feat_ext(self, x):

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)

        xf = self.backbone.avgpool(x4)
        xf = xf.view(xf.size(0), -1)

        return xf

    def fwd_pass_and_loss(self, batch):
        ims = batch["images"]
        bbs = batch["bbs"]
        j2ds = batch["j2d"]
        cam_intr = batch["cam_intr"]
        moshed_gt = batch["moshpose"]
        mosh_available = batch["mosh_available"]
        batch_size = j2ds.shape[0]
        num_cams = j2ds.shape[1]
        seq_size = j2ds.shape[2]

        # Koila lazy tensor
        # (ims,bbs,j2ds,cam_intr,moshed_gt) = lazy(ims,bbs,j2ds,cam_intr,moshed_gt)


        # Network forward
        cam_poses, pred_pose_smpl, pred_betas, pred_motion_z = self.forward(ims, bbs, trans_scale=0.05)

        ############## Sanity check ###################
        # cam_poses = torch.cat([self.init_position.expand(batch_size,num_cams,seq_size, -1), 
        #                 self.init_orient.expand(batch_size,num_cams,seq_size,-1)],3)
        # pred_pose = self.mvae_model.decode(2*torch.randn(1,1024).type_as(pred_motion_z))
        # pred_pose_unnorm = pred_pose*self.mvae_std + self.mvae_mean
        # pred_pose_smpl = mop2smpl(pred_pose_unnorm.reshape(batch_size*seq_size,22,9),self.smpl).reshape(batch_size,seq_size,-1)
        # pred_betas = self.init_shape.expand(batch_size,-1)
        # pred_pose_smpl[:,:,6:] = moshed_gt
        ###############################################
        
        pred_pose_smpl_reshaped = pred_pose_smpl.reshape(-1, pred_pose_smpl.shape[-1])
        
        # SMPL forward
        smpl_out = self.smpl.forward(root_orient = pred_pose_smpl_reshaped[:,3:6],
                            pose_body = pred_pose_smpl_reshaped[:,6:],
                            trans = pred_pose_smpl_reshaped[:,:3],
                            betas = pred_betas.unsqueeze(1).expand(-1,seq_size,-1).reshape(-1,pred_betas.shape[-1]))
        
        # j3ds = torch.einsum("ij,bjk->bik",self.j_regr.squeeze(0),smpl_out.v)
        j3ds = smpl_out.Jtr[:,:22,:]
        
        
        # camera projection
        proj_j3ds = torch.stack([perspective_projection(j3ds,
                        p3d_rt.rotation_6d_to_matrix(cam_poses[:,i,:,3:].reshape(-1,6)),
                        cam_poses[:,i,:,:3].reshape(-1,3),
                        cam_intr[:,i].unsqueeze(1).expand(-1,seq_size,-1,-1).reshape(-1,3,3).float()).reshape(batch_size,seq_size,-1,2) for i in range(num_cams)]).permute(1,0,2,3,4)
        

        # reprojection loss
        loss_2d = (j2ds[:,:,:,:22,2]*(((proj_j3ds[:,:,:,:22] - j2ds[:,:,:,:22,:2])**2).sum(dim=4))).mean()
        
        # latent space regularization
        loss_z = (pred_motion_z*pred_motion_z).mean()

        # smooth camera motions
        loss_cams = ((cam_poses[:,:,1:] - cam_poses[:,:,:-1])**2).mean()

        # shape regularization loss
        loss_betas = (pred_betas*pred_betas).mean()

        # Articulated pose loss using moshed GT
        try:
            if mosh_available.any():
                theta_loss = ((p3d_rt.matrix_to_rotation_6d(p3d_rt.axis_angle_to_matrix(pred_pose_smpl[mosh_available,:,6:].reshape(-1,3))) - 
                            p3d_rt.matrix_to_rotation_6d(p3d_rt.axis_angle_to_matrix(moshed_gt[mosh_available].reshape(-1,3))))**2).mean()
            else:
                theta_loss = torch.sum(torch.zeros(1).type_as(loss_2d))
        except:
            theta_loss = torch.sum(torch.zeros(1).type_as(loss_2d))
            print("Mosh not available")

        # VPoser loss on the output
        self.vp_model.eval()
        vp_sample = self.vp_model.encode(pred_pose_smpl_reshaped[:,6:]).mean
        loss_vposer = (vp_sample*vp_sample).mean()

        # total loss
        loss = self.hparams["train_loss_2d_weight"] * loss_2d + \
                self.hparams["train_loss_z_weight"] * loss_z + \
                    self.hparams["train_loss_cams_weight"] * loss_cams + \
                        self.hparams["train_loss_betas_weight"] * loss_betas +\
                            self.hparams["train_loss_theta_weight"] * theta_loss +\
                                self.hparams["train_loss_vposer_weight"] * loss_vposer

        losses = {"loss_2d": loss_2d.detach().cpu().numpy(),
                    "loss_z": loss_z.detach().cpu().numpy(),
                    "loss_cams": loss_cams.detach().cpu().numpy(),
                    "loss_betas": loss_betas.detach().cpu().numpy(),
                    "loss_theta": theta_loss.detach().cpu().numpy(),
                    "loss_vposer": loss_vposer.detach().cpu().numpy()}

        cam_poses_actual = torch.cat([p3d_rt.rotation_6d_to_matrix(cam_poses[:,:,:,3:]),cam_poses[:,:,:,:3].unsqueeze(4)],dim=4)
        cam_poses_actual = torch.cat([cam_poses_actual,torch.tensor([0,0,0,1]).type_as(cam_poses_actual).repeat(batch_size,num_cams,seq_size,1,1)],dim=3)
        cam_poses_actual = torch.inverse(cam_poses_actual)
        output = {"smpl_out_v": smpl_out.v.detach(),
                    "pred_pose": pred_pose_smpl_reshaped.detach(),
                    "pred_shape": pred_betas.detach(),
                    "cam_poses": cam_poses.detach(),
                    "cam_trans": cam_poses_actual[:,:,:,:3,3].detach(),
                    "cam_rots": p3d_rt.matrix_to_quaternion(cam_poses_actual[:,:,:,:3,:3]).detach(),
                    "cam_intr": cam_intr,
                    "j2ds": j2ds.detach()}

        
        return output, losses, loss

    def training_step(self, batch, batch_idx):
        output, losses, loss = self.fwd_pass_and_loss(batch)

        return {"output":output, "loss":loss, "losses":losses, "batch_images": batch["full_im_paths"]}


    def validation_step(self, batch, batch_idx):
        output, losses, loss = self.fwd_pass_and_loss(batch)
        
        return {"val_losses":losses,"val_loss":loss, "output": output, "batch_images": batch["full_im_paths"]}

    def test_step(self, batch, batch_idx):
        output, losses, loss = self.fwd_pass_and_loss(batch)

        return {"output":output, "loss":loss, "losses":losses, "batch_images": batch["full_im_paths"]}

    
    def training_epoch_end(self, outputs):
        for loss_name, val in outputs[0]["losses"].items():
            val_list = []
            for x in outputs:
                val_list.append(x["losses"][loss_name])
            mean_val = np.mean(val_list)
            self.logger.experiment.add_scalar(loss_name + '/train', mean_val, self.global_step)
        self.log("loss", np.mean([x["loss"].detach().cpu().numpy() for x in outputs]))

        # random index for viz
        idx = np.random.randint(len(outputs))
        viz_images = self.seq_smpl_proj(outputs[idx])
        if not os.path.exists(ospj(self.logger.log_dir,"viz_gifs")):
            os.makedirs(ospj(self.logger.log_dir,"viz_gifs"))

        if self.current_epoch % self.hparams["train_check_val_every_n_epoch"] == 0:
            np.savez(ospj(self.logger.log_dir,"viz_gifs",str(self.current_epoch)+"_train"),
                verts=outputs[idx]["output"]["smpl_out_v"].detach().cpu().numpy(),
                cam_trans=outputs[idx]["output"]["cam_trans"].detach().cpu().numpy(),
                cam_rots=outputs[idx]["output"]["cam_rots"].detach().cpu().numpy())
        
        imageio.mimsave(ospj(self.logger.log_dir,"viz_gifs",str(self.current_epoch)+"_train.gif"),
                    [make_grid(torch.from_numpy(viz_images[:,i]).permute(0,3,1,2),nrow=viz_images.shape[0]).permute(1,2,0).cpu().numpy()[::5,::5] for i in range(viz_images.shape[1])])

        

    def validation_epoch_end(self, outputs):
        for loss_name, val in outputs[0]["val_losses"].items():
            val_list = []
            for x in outputs:
                val_list.append(x["val_losses"][loss_name])
            mean_val = np.mean(val_list)
            self.logger.experiment.add_scalar(loss_name + '/val', mean_val, self.global_step)
        self.log("val_loss", np.mean([x["val_loss"].detach().cpu().numpy() for x in outputs]))

        # random index for viz
        idx = np.random.randint(len(outputs))
        viz_images = self.seq_smpl_proj(outputs[idx])
        if not os.path.exists(ospj(self.logger.log_dir,"viz_gifs")):
            os.makedirs(ospj(self.logger.log_dir,"viz_gifs"))

        np.savez(ospj(self.logger.log_dir,"viz_gifs",str(self.current_epoch)+"_val"),
             verts=outputs[idx]["output"]["smpl_out_v"].detach().cpu().numpy(),
             cam_trans=outputs[idx]["output"]["cam_trans"].detach().cpu().numpy(),
             cam_rots=outputs[idx]["output"]["cam_rots"].detach().cpu().numpy())
        
        imageio.mimsave(ospj(self.logger.log_dir,"viz_gifs",str(self.current_epoch)+"_val.gif"),
                    [make_grid(torch.from_numpy(viz_images[:,i]).permute(0,3,1,2),nrow=viz_images.shape[0]).permute(1,2,0).cpu().numpy()[::5,::5] for i in range(viz_images.shape[1])])


    def test_epoch_end(self, outputs):
        
        import ipdb;ipdb.set_trace()
        
        # random index for viz
        idx = np.random.randint(len(outputs))
        viz_images = self.seq_smpl_proj(outputs[idx])
        if not os.path.exists(ospj(self.logger.log_dir,"viz_gifs")):
            os.makedirs(ospj(self.logger.log_dir,"viz_gifs"))

        imageio.mimsave(ospj(self.logger.log_dir,"viz_gifs",str(self.current_epoch)+"_train.gif"),
                    [make_grid(torch.from_numpy(viz_images[:,i]).permute(0,3,1,2),nrow=viz_images.shape[0]).permute(1,2,0).cpu().numpy()[::5,::5] for i in range(viz_images.shape[1])])



    def seq_smpl_proj(self,output):

        full_images_paths = output["batch_images"]
        num_cams = len(full_images_paths)
        seq_len = len(full_images_paths[0])
        if self.hparams["data_name"].lower() == "h36m":
            im_res = [1000,1000]
        # random index
        idx = np.random.randint(output["output"]["cam_poses"].shape[0])
        rend_ims = np.zeros([num_cams,seq_len,im_res[0],im_res[1],3])
        for cam in range(num_cams):
            for s in range(seq_len):
                im = cv2.imread(full_images_paths[cam][s][idx])[:im_res[0],:im_res[1],::-1]/255.
                rend_ims[cam,s] = self.renderer(output["output"]["smpl_out_v"].view(-1,seq_len,6890,3)[idx,s].detach().cpu().numpy(),
                                    output["output"]["cam_poses"][idx,cam,s,:3].detach().cpu().numpy(),
                                    p3d_rt.rotation_6d_to_matrix(output["output"]["cam_poses"][idx,cam,s,3:].unsqueeze(0)).detach().cpu().numpy(),
                                    im,intr=output["output"]["cam_intr"][idx,cam].detach().cpu().numpy(),
                                    faces=self.smpl.f.detach().cpu().numpy())

        return rend_ims
        
        


    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        if self.hparams["train_train_reg_only"]:
            print("training regressor only")
            for p in self.backbone.parameters():
                p.requires_grad = False
            
        for p in self.vp_model.parameters():
            p.requires_grad = False

        if self.hparams["model_zspace"].lower() == "motion_vae":
            for p in self.mvae_model.parameters():
                p.requires_grad = False
        
        params = list(self.parameters())
        new_params=[]
        for p in params:
            if p.requires_grad:
                new_params.append(p)
        
        optimizer = torch.optim.Adam(new_params, 
                                lr=self.hparams["train_lr"],
                                weight_decay=0,
                                amsgrad=True)

        return optimizer


    def train_dataloader(self):
        if self.hparams.data_name.lower() == "h36m":
            train_dset = h36m.h36m(self.hparams,subjects=["S1","S5"])
        else:
            import ipdb;ipdb.set_trace()

        return DataLoader(train_dset, batch_size=self.hparams.train_batch_size,
                            num_workers=self.hparams.train_num_workers,
                            shuffle=self.hparams.train_shuffle_dset,
                            drop_last=True)

    def val_dataloader(self):
        if self.hparams.data_name.lower() == "h36m":
            val_dset = h36m.h36m(self.hparams,subjects=["S6"])
        else:
            import ipdb;ipdb.set_trace()

        return DataLoader(val_dset, batch_size=self.hparams.train_val_batch_size,
                            num_workers=self.hparams.train_num_workers,
                            shuffle=self.hparams.train_shuffle_dset,
                            drop_last=True)



        





# python src/zcopenet/trainer.py --name=test --version=0 --model=copenet_twoview --datapath=/home/nsaini/Datasets/AerialPeople/agora_copenet_uniform_new_cropped/