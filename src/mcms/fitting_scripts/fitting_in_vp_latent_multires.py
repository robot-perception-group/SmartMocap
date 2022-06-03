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


class mcms_vp_fitting_multires():
    def __init__(self,config_path):
        super().__init__()
        self.config = yaml.safe_load(open(config_path))

        # config parameters
        self.batch_size = self.config['batch_size']
        self.seq_len = self.config["seq_len"]
        self.loss_2d_weight = self.config["loss_2d_weight"]
        self.loss_z_weight = self.config["loss_z_weight"]
        self.loss_cams_weight = self.config["loss_cams_weight"]
        self.loss_betas_weight = self.config["loss_betas_weight"]
        self.n_optim_iters = self.config["n_optim_iters"]
        self.loss_human_gp_weight = self.config["loss_human_gp_weight"]
        self.loss_cam_gp_weight = self.config["loss_cam_gp_weight"]
        self.loss_vp_weight = self.config["loss_vp_weight"]
        self.loss_smpl_in_front_weight = self.config["loss_smpl_in_front_weight"]
        self.lr = self.config["lr"]
        self.dset = self.config["dset"]
        self.seq_no = self.config["seq_no"]
        self.big_seq_start = self.config["big_seq_start"]
        self.big_seq_end = self.config["big_seq_end"]
        self.overlap = self.config["overlap"]
        self.hparams = yaml.safe_load(open("/".join(self.config["mo_vae_ckpt_path"].split("/")[:-2])+"/hparams.yaml","r"))
        self.device = torch.device(self.config["device"])

        # SMPL
        self.smpl = BodyModel(bm_fname=self.hparams["model_smpl_neutral_path"]).to(self.device)

        # Motion VAE
        self.nmg_hparams = yaml.safe_load(open("/".join(self.hparams["train_motion_vae_ckpt_path"].split("/")[:-2])+"/hparams.yaml"))
        self.mvae_model = nmg.nmg.load_from_checkpoint(self.hparams["train_motion_vae_ckpt_path"],self.nmg_hparams).to(self.device)
        mean_std = np.load(self.hparams["model_mvae_mean_std_path"])
        self.mvae_mean = torch.from_numpy(mean_std["mean"]).float().to(self.device)
        self.mvae_std = torch.from_numpy(mean_std["std"]).float().to(self.device)

        # vposer model
        self.vp_model = load_model(self.hparams["model_vposer_path"], model_code=VPoser,remove_words_in_model_weights="vp_model.")[0].to(self.device)
        self.vp_model.eval()

        # Dataloader
        if self.dset.lower() == "h36m":
            self.ds = h36m.h36m(self.hparams,used_cams=self.config["cams_used"])
            self.fps_scl = 2
            self.viz_dwnsample = 1
            # renderer
            self.im_res = [[1000,1000],[1000,1000],[1000,1000],[1000,1000]]
        elif self.dset.lower() == "copenet_real":
            self.hparams["data_datapath"] = "/home/nsaini/Datasets/copenet_data"
            self.ds = copenet_real.copenet_real(self.hparams,range(0,7000),used_cams=self.config["cams_used"])
            self.fps_scl = 1
            self.viz_dwnsample = 1
            self.im_res=[[1920,1080],[1920,1080]]
        elif self.dset.lower() == "savitr":
            self.hparams["data_datapath"] = self.config["data_path"]
            self.ds = savitr_dataset.savitr_dataset(self.hparams["data_datapath"],seq_len=25,used_cams=self.config["cams_used"])
            im_lists = self.ds.__getitem__(0,seq_len=1)["full_im_paths"]
            self.im_res = [[cv2.imread(i[0]).shape[1],cv2.imread(i[0]).shape[0]] for i in im_lists]
            self.fps_scl = 1
            self.viz_dwnsample = 1
        elif self.dset.lower() == "rich":
            self.hparams["data_datapath"] = "/ps/project/datasets/AirCap_ICCV19/RICH_IPMAN/test/2021-06-15_Multi_IOI_ID_00186_Yoga1"
            self.ds = rich.rich(self.hparams["data_datapath"],used_cams=self.config["cams_used"])
            self.im_res = [[4112,3008],[4112,3008],[4112,3008],[3008,4112],[4112,3008],[3008,4112],[4112,3008],[4112,3008]]
            self.fps_scl = 1
            self.viz_dwnsample = 1
        self.renderer = [Renderer(img_res=[np.ceil(res[0]/self.viz_dwnsample),np.ceil(res[1]/self.viz_dwnsample)]) for res in self.im_res]

        # make dir for seq_no
        if self.dset.lower() == "h36m":
            os.makedirs("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}".format(self.config["trial_name"],self.seq_no),exist_ok=True)
        else:
            self.seq_no = 0
            os.makedirs("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}".format(self.config["trial_name"],self.seq_no),exist_ok=True)


    def load_batch_wth_pare_init(self):
        # dump yaml file
        yaml.safe_dump(self.config,open("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/config.yml".format(self.config["trial_name"],self.seq_no),"w"))

        self.init_cam_position = []
        self.init_cam_orient = []
        self.init_smpl_trans_rf = []
        self.init_smpl_orient_rf = []
        self.init_smpl_trans_ff = []
        self.init_smpl_orient_ff = []
        self.init_smpl_art_motion_vp_latent = []
        self.init_smpl_shape = []
        self.pare_init_state = []

        self.init_stage_j2d = []
        self.init_stage_im_paths = []
        with trange(self.big_seq_start,self.big_seq_end,self.fps_scl*(self.seq_len-self.overlap)) as seq_t:
            for seq_start in seq_t:

                # tensorboard summarywriter
                os.makedirs("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/{:04d}".format(self.config["trial_name"],self.seq_no,seq_start),exist_ok=True)
                self.writer = SummaryWriter(log_dir="/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/{:04d}".format(self.config["trial_name"],self.seq_no,seq_start), 
                                filename_suffix="{:04d}".format(seq_start))

                if self.dset.lower() == "h36m":
                    # get batch
                    batch = self.ds.__getitem__(self.seq_no,seq_start)
                else:
                    batch = self.ds.__getitem__(seq_start)
                j2ds = batch["j2d"].float().to(self.device)
                self.init_stage_j2d.append(j2ds.detach())
                self.init_stage_im_paths.append(batch["full_im_paths"])
                self.num_cams = j2ds.shape[0]
                self.cam_intr = torch.from_numpy(batch["cam_intr"]).float().to(self.device)

                cam_position = []
                cam_orient = []
                for c in range(len(self.config["cams_used"])):
                    if c in self.config["position_changing_cams"]:
                        cam_position.append(torch.tensor([0,0,5]).float().repeat(1,self.seq_len,1).to(self.device))
                    else:
                        cam_position.append(torch.tensor([0,0,5]).float().repeat(1,1,1).to(self.device))
                
                smpl_shape = torch.zeros(10).unsqueeze(0).to(self.device)

                self.mvae_model.eval()
                smpl_motion_init = nmg2smpl((self.mvae_model.decode(torch.zeros(1,1024).to(self.device))*self.mvae_std + self.mvae_mean).reshape(self.seq_len,22,9),self.smpl)
                smpl_trans = smpl_motion_init[:,:3].detach()
                smpl_orient = p3d_rt.axis_angle_to_matrix(smpl_motion_init[:,3:6])
                smpl_art_motion_vp_latent_init = self.vp_model.encode(batch["pare_poses"][:, :21].reshape(25, 63).to(self.device)).mean.detach()
                # first frame init
                pare_orient = p3d_rt.axis_angle_to_matrix(batch["pare_orient"].to(self.device))
                cam_orient = torch.matmul(pare_orient[:,0],torch.inverse(smpl_orient[0:1])).unsqueeze(1)
                cam_orient = [cam_orient[c].detach().unsqueeze(0).repeat([1,self.seq_len,1,1]) if c in self.config["orient_changing_cams"] 
                                else cam_orient[c,0:1].detach().unsqueeze(0) for c in range(self.num_cams)]
                smpl_orient = smpl_orient.detach()

                self.pare_init_state.append({"cam_orient":cam_orient, "cam_position": cam_position,
                            "smpl_trans":smpl_trans, 
                            "smpl_orient": smpl_orient, "smpl_shape": smpl_shape, 
                            "smpl_art_motion_vp_latent":smpl_art_motion_vp_latent_init, "j2ds":j2ds.detach(), 
                            "full_im_paths": batch["full_im_paths"]})

        return self.pare_init_state

    def optimize(self, params):
        
        cam_orient = [p3d_rt.matrix_to_rotation_6d(x).detach().requires_grad_(True) for x in params["cam_orient"]]
        cam_position = [x.detach().requires_grad_(True) for x in params["cam_position"]]
        smpl_trans_ff = params["smpl_trans"][:1].detach().requires_grad_(True)
        smpl_orient_ff = p3d_rt.matrix_to_rotation_6d(params["smpl_orient"][:1]).detach().requires_grad_(True)
        smpl_trans_rf = params["smpl_trans"][1:].detach().requires_grad_(True)
        smpl_orient_rf = p3d_rt.matrix_to_rotation_6d(params["smpl_orient"][1:]).detach().requires_grad_(True)
        smpl_shape = params["smpl_shape"].detach().requires_grad_(True)
        smpl_art_motion_vp_latent = params["smpl_art_motion_vp_latent"].detach().requires_grad_(True)
        j2ds = params["j2ds"].detach()
        curr_seq_len = j2ds.shape[1]

        optim0 = torch.optim.Adam(cam_orient + cam_position,lr=self.lr)
        optim1 = torch.optim.Adam(cam_orient + cam_position + [smpl_trans_rf,smpl_orient_rf],lr=self.lr)
        optim2 = torch.optim.Adam(cam_orient + cam_position + [smpl_art_motion_vp_latent, smpl_trans_rf, smpl_orient_rf, smpl_shape],lr=self.lr)

        with trange(self.n_optim_iters) as t:
            for i in t:
                
                if i < self.config["n_optim_iters_stage0"]:
                    stage = 0
                    optim = optim0
                elif i > self.config["n_optim_iters_stage0"] and i < self.config["n_optim_iters_stage1"]:
                    stage = 1
                    optim = optim1
                else:
                    stage = 2
                    optim = optim2

                ################### FWD pass #####################
                # smpl_art_motion_interm = torch.cat([smpl_art_motion[:,:9],smpl_art_motion_nonOpt[:,:3],
                #                                     smpl_art_motion[:,9:11],smpl_art_motion_nonOpt[:,3:],
                #                                     smpl_art_motion[:,11:]],dim=1)
                smpl_art_motion_interm = self.vp_model.decode(smpl_art_motion_vp_latent)["pose_body"]
                smpl_trans = torch.cat([smpl_trans_ff,smpl_trans_rf])
                smpl_orient = torch.cat([smpl_orient_ff,smpl_orient_rf])
                
                smpl_motion = torch.cat([smpl_trans.unsqueeze(1),p3d_rt.matrix_to_axis_angle(p3d_rt.rotation_6d_to_matrix(smpl_orient)).unsqueeze(1),
                                        smpl_art_motion_interm],dim=1).reshape(curr_seq_len,69)
                # Decode smpl motion using motion vae
                self.mvae_model.eval()
                smpl_motion_latent = []
                for strt_idx in range(0,curr_seq_len-self.seq_len+1,2):
                    curr_pos, curr_ori = geometry.get_ground_point(smpl_trans[strt_idx],p3d_rt.rotation_6d_to_matrix(smpl_orient[strt_idx]))
                    curr_tfm = to_homogeneous(curr_ori,curr_pos)
                    curr_root_pose = to_homogeneous(p3d_rt.rotation_6d_to_matrix(smpl_orient[strt_idx:strt_idx+self.seq_len]),smpl_trans[strt_idx:strt_idx+25])
                    canonical_root_pose = torch.matmul(torch.inverse(curr_tfm),curr_root_pose)
                    canonical_smpl_motion = torch.cat([canonical_root_pose[:,:3,3],
                                            p3d_rt.matrix_to_axis_angle(canonical_root_pose[:,:3,:3]),
                                            smpl_motion[strt_idx:strt_idx+self.seq_len,6:]],dim=1)
                    nmg_repr = (smpl2nmg(canonical_smpl_motion,self.smpl).reshape(-1,25,22*9) - self.mvae_mean)/self.mvae_std
                    smpl_motion_latent.append(self.mvae_model.encode(nmg_repr)[:,0])
                # nmg_repr = (smpl2nmg(smpl_motion,self.smpl).reshape(-1,curr_seq_len,22*9) - self.mvae_mean)/self.mvae_std
                # smpl_motion_latent = self.mvae_model.encode(nmg_repr)[:,0]

                # SMPL fwd pass
                smpl_out = self.smpl.forward(root_orient = smpl_motion[:,3:6],
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
                cam_ext = torch.cat([cam_ext_temp,torch.tensor([0,0,0,1]).type_as(cam_ext_temp).repeat(self.num_cams,curr_seq_len,1,1)],dim=2)

                # camera extrinsics
                cam_poses = torch.inverse(cam_ext)

                # camera projection
                proj_j3ds = torch.stack([perspective_projection(j3ds,
                                cam_ext[i,:,:3,:3].reshape(-1,3,3),
                                cam_ext[i,:,:3,3].reshape(-1,3),
                                self.cam_intr[i].unsqueeze(0).expand(curr_seq_len,-1,-1).reshape(-1,3,3)).reshape(curr_seq_len,-1,2) for i in range(self.num_cams)])


                ####################### Losses #######################

                # reprojection loss
                idcs = torch.where(j2ds[0,0,:22,2]!=0)[0]

                if stage == 0:
                    loss_2d = (j2ds[:,:,idcs,2]*(((proj_j3ds[:,:,idcs] - j2ds[:,:,idcs,:2])**2).sum(dim=3))).sum(dim=0).mean()
                # elif stage == 1:
                elif False:
                    loss_2d = (j2ds[:,:,idcs,2]*((gmcclure(proj_j3ds[:,:,idcs] - j2ds[:,:,idcs,:2],config["gmcclure_sigma"])).sum(dim=3))).sum(dim=0).mean()
                else:
                    loss_2d = (j2ds[:,:,idcs,2]*(((proj_j3ds[:,:,idcs] - j2ds[:,:,idcs,:2])**2).sum(dim=3))).sum(dim=0).mean()

                # latent space regularization
                loss_z = torch.cat([(x*x).mean().unsqueeze(0) for x in smpl_motion_latent]).sum()

                # smooth camera motions
                loss_cams = ((cam_poses[:,1:] - cam_poses[:,:-1])**2).mean()

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
                loss = self.loss_2d_weight[stage] * loss_2d + \
                        self.loss_z_weight[stage] * loss_z + \
                            self.loss_cams_weight[stage] * loss_cams + \
                                self.loss_betas_weight[stage] * loss_betas + \
                                    float(self.loss_human_gp_weight[stage]) * loss_human_gp + \
                                        float(self.loss_cam_gp_weight[stage]) * loss_cam_gp + \
                                            self.loss_vp_weight[stage] * loss_vp + \
                                                self.loss_smpl_in_front_weight[stage] * loss_smpl_in_front

                # Viz

                # zero grad optimizer
                optim.zero_grad()
                
                # backward
                loss.backward()

                # optim step
                optim.step()

                # loss dict
                loss_dict = {"loss":loss.item(),
                                "loss_2d":loss_2d.item(),
                                "loss_z":loss_z.item(),
                                "loss_cams":loss_cams.item(),
                                "loss_betas":loss_betas.item(),
                                "loss_human_gp":loss_human_gp.item(),
                                "loss_cam_gp":loss_cam_gp.item(),
                                "loss_vp":loss_vp.item(),
                                "loss_smpl_front":loss_smpl_in_front.item()}

                # print loss
                t.set_postfix(loss_dict)

                # track losses
                for k,v in loss_dict.items():
                    self.writer.add_scalar(k,v,global_step=i)
        
        return {"cam_orient":[p3d_rt.rotation_6d_to_matrix(x).detach() for x in cam_orient], "cam_position": [x.detach() for x in cam_position],
                            "smpl_trans":smpl_trans.detach(), 
                            "smpl_orient": p3d_rt.rotation_6d_to_matrix(smpl_orient), "smpl_shape": smpl_shape, 
                            "smpl_art_motion_vp_latent":smpl_art_motion_vp_latent.detach(), "j2ds":j2ds.detach(),
                            "full_im_paths": params["full_im_paths"]}



    def stitch(self,prev_stage_dicts,stitch_num):
        
        new_stage_dicts = []

        for n_dict in range(0,len(prev_stage_dicts)-stitch_num+1,stitch_num):
            prev_dict = prev_stage_dicts[n_dict]
            
            cam_orient = prev_dict["cam_orient"]
            cam_position = prev_dict["cam_position"]
            smpl_trans = prev_dict["smpl_trans"]
            smpl_shape = prev_dict["smpl_shape"]
            smpl_orient = prev_dict["smpl_orient"]
            smpl_art_motion_vp_latent = prev_dict["smpl_art_motion_vp_latent"]
            last_smpl_orient = prev_dict["smpl_orient"].detach()
            last_smpl_trans = prev_dict["smpl_trans"].detach()
            j2ds = prev_dict["j2ds"]
            full_im_paths = prev_dict["full_im_paths"]
            
            for i in range(1,stitch_num):
                prev_dict = prev_stage_dicts[n_dict+i]
                pos, ori = geometry.get_ground_point(last_smpl_trans[-self.overlap+1],last_smpl_orient[-self.overlap+1])
                prev_tfm = to_homogeneous(ori,pos)
                curr_pos, curr_ori = geometry.get_ground_point(prev_dict["smpl_trans"][0],prev_dict["smpl_orient"][0])
                curr_tfm = to_homogeneous(curr_ori,curr_pos)
                tfm = torch.matmul(prev_tfm,torch.inverse(curr_tfm))
                curr_smpl = to_homogeneous(prev_dict["smpl_orient"],prev_dict["smpl_trans"])
                tfmd_smpl = torch.matmul(tfm,curr_smpl)
                
                smpl_orient = torch.cat([smpl_orient[:-self.overlap], tfmd_smpl[:,:3,:3]])
                smpl_trans = torch.cat([smpl_trans[:-self.overlap],tfmd_smpl[:,:3,3]]).detach()

                smpl_art_motion_vp_latent = torch.cat([smpl_art_motion_vp_latent[:-self.overlap],prev_dict["smpl_art_motion_vp_latent"]])

                updated_cam_orient = [torch.matmul(tfm[:,:3,:3],x[0,:,:3,:3]) for x in prev_dict["cam_orient"]]
                updated_cam_position = [torch.matmul(tfm[:,:3,:3],x[0,:].unsqueeze(-1)).squeeze(-1) + tfm[0,:3,3] for x in prev_dict["cam_position"]]
                
                for j in range(len(cam_orient)):
                    if cam_orient[j].shape[1] != 1:
                        cam_orient[j] = torch.cat([cam_orient[j][:,:-self.overlap], updated_cam_orient[j].unsqueeze(0)],dim=1)
                        cam_position[j] = torch.cat([cam_position[j][:,:-self.overlap], updated_cam_position[j].unsqueeze(0)],dim=1)

                # save last smpl orient
                last_smpl_orient = tfmd_smpl[:,:3,:3].detach()
                last_smpl_trans = tfmd_smpl[:,:3,3].detach()

                j2ds = torch.cat([j2ds[:,:-self.overlap],prev_dict["j2ds"].detach()],dim=1)

                for k in range(len(full_im_paths)):
                    full_im_paths[k] = full_im_paths[k][:-self.overlap] + prev_dict["full_im_paths"][k]
                

            new_stage_dicts.append({"cam_orient":cam_orient, "cam_position": cam_position,
                            "smpl_trans":smpl_trans.detach(), 
                            "smpl_orient": smpl_orient, "smpl_shape": smpl_shape, 
                            "smpl_art_motion_vp_latent":smpl_art_motion_vp_latent.detach(), "j2ds":j2ds.detach(),
                            "full_im_paths": full_im_paths})

        return new_stage_dicts

                

    def save_results(self,stage_dict,stage,seq_start):

        os.makedirs("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/stage_{:02d}_seq_start_{:05d}".format(self.config["trial_name"],
                                    self.seq_no,stage,seq_start),exist_ok=True)

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
        cam_ext = torch.cat([cam_ext_temp,torch.tensor([0,0,0,1]).type_as(cam_ext_temp).repeat(self.num_cams,curr_seq_len,1,1)],dim=2)
        cam_poses = torch.inverse(cam_ext)
        
        smpl_art_motion_interm = self.vp_model.decode(smpl_art_motion_vp_latent)["pose_body"]
        smpl_motion = torch.cat([smpl_trans.unsqueeze(1),p3d_rt.matrix_to_axis_angle(smpl_orient).unsqueeze(1),
                                        smpl_art_motion_interm],dim=1).reshape(curr_seq_len,69)

        # SMPL fwd pass
        smpl_out = self.smpl.forward(root_orient = smpl_motion[:,3:6],
                                    pose_body = smpl_motion[:,6:],
                                    trans = smpl_motion[:,:3],
                                    betas = smpl_shape.unsqueeze(1).expand(-1,curr_seq_len,-1).reshape(-1,smpl_shape.shape[-1]))


        rend_ims = [np.zeros([curr_seq_len,np.ceil(self.im_res[c][1]).astype(int),np.ceil(self.im_res[c][0]).astype(int),3]) for c in range(self.num_cams)]
        for cam in tqdm(range(self.num_cams)):
            os.makedirs("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/seq_no_{:04d}/stage_{:02d}/seq_start_{:05d}/cam_{:02d}".format(self.config["trial_name"],
                                    self.seq_no,stage,seq_start,cam),exist_ok=True)
            for s in range(curr_seq_len):
                im = cv2.imread(full_images_paths[cam][s])[:self.im_res[cam][1],:self.im_res[cam][0],::-1]/255.
                temp_intr = self.cam_intr[cam].detach().cpu().numpy()
                temp_v = smpl_out.v.view(curr_seq_len,6890,3)[s].detach().cpu().numpy()
                temp_v[:,2] = temp_v[:,2]
                
                rend_ims[cam][s] = self.renderer[cam](temp_v,
                                    cam_ext[cam,s,:3,3].detach().cpu().numpy(),
                                    cam_ext[cam,s,:3,:3].unsqueeze(0).detach().cpu().numpy(),
                                    im,intr=temp_intr,
                                    faces=self.smpl.f.detach().cpu().numpy())
                for joint in range(j2ds.shape[3]):
                    rend_ims[cam][s] = rend_ims[cam][s]*255
                    cv2.circle(rend_ims[cam][s],(int(j2ds[cam,s,joint,0]),
                                int(j2ds[cam,s,joint,1])),10,(255,255,255),-1)
                    rend_ims[cam][s] = rend_ims[cam][s]/255
                
                cv2.imwrite("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/seq_no_{:04d}/stage_{:02d}/seq_start_{:05d}/cam_{:02d}/{:05d}.png".format(self.config["trial_name"],
                                    self.seq_no,stage,seq_start,cam,seq_start+s),rend_ims[cam][s,::5,::5,::-1]*255)
        # for c in range(self.num_cams):
        #     imageio.mimsave("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/test_{:05d}_cam_{:02d}_itr_{:04d}.gif".format(self.config["trial_name"],
        #                 self.seq_no,stage,seq_start,c),list(rend_ims[c][:,::2,::2]))
            # imageio.mimsave("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/test_{:05d}_cam_{:02d}.gif".format(config["trial_name"],seq_no,seq_start,c),
            #             [make_grid(torch.from_numpy(rend_ims[:,i]).permute(0,3,1,2),
            #                 nrow=rend_ims.shape[0]).permute(1,2,0).cpu().numpy()[::5,::5] for i in range(rend_ims.shape[1])])


        np.savez("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/seq_no_{:04d}/stage_{:02d}/seq_start_{:05d}".format(self.config["trial_name"],
                                    self.seq_no,stage,seq_start),
            verts=smpl_out.v.detach().cpu().numpy(),
            cam_trans=cam_poses[:,:,:3,3].detach().cpu().numpy(),
            cam_rots=p3d_rt.matrix_to_quaternion(cam_poses[:,:,:3,:3]).detach().cpu().numpy())




if __name__ == "__main__":
    fitter = mcms_vp_fitting_multires("/is/ps3/nsaini/projects/mcms/src/mcms/fitting_scripts/fit_config.yml")
    fitter.load_batch_wth_pare_init()
    
    fitter.res_stage = []
    
    stage = 0
    
    res_stage = Parallel(n_jobs=-1,prefer="threads")(delayed(fitter.optimize)(fitter,fitter.pare_init_state[i]) for i in tqdm(range(len(fitter.init_stage_j2d))))
    # res_stage = []
    # for i in tqdm(range(len(fitter.init_stage_j2d))):
    #     res_stage.append(fitter.optimize(fitter.pare_init_state[i]))
    fitter.res_stage.append(res_stage)
    stitched_res = fitter.stitch(fitter.res_stage[-1],2)
    stage = 1

    print("\n")
    print("############### stage length {} ##############".format(len(res_stage)))
    print("\n")

    while len(res_stage) != 1:
        # res_stage = []
        # for i in tqdm(range(len(stitched_res))):
        #     res_stage.append(fitter.optimize(stitched_res[i]))
        res_stage = Parallel(n_jobs=-1,prefer="threads")(delayed(fitter.optimize)(fitter,stitched_res[i]) for i in tqdm(range(len(stitched_res))))
        fitter.res_stage.append(res_stage)
        stitched_res = fitter.stitch(fitter.res_stage[-1],2)
        print("\n")
        print("############### stage length {} ##############".format(len(res_stage)))
        print("\n")
        stage = stage+1
    
    print("##### saving results ######")
    fitter.save_results(res_stage[0],stage,fitter.big_seq_start)
    

# # Full fittings
# full_cam_orient = []
# full_cam_position = []
# full_smpl_verts = []
# full_smpl_shape = []
# full_smpl_motion_latent = []
# full_vp_latent = []
# full_cam_ext = []
# full_smpl_orient = []
# full_smpl_trans = []
# full_smpl_motion = []
# full_j2d = []
# # overlay_gifs = []

# # make dir for seq_no
# if dset.lower() == "h36m":
#     os.makedirs("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}".format(config["trial_name"],seq_no),exist_ok=True)
# else:
#     seq_no = 0
#     os.makedirs("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}".format(config["trial_name"],seq_no),exist_ok=True)


# # Camera and SMPL params
# # cam_position_staticcam = torch.tensor([0,0,5]).float().repeat(batch_size,num_cams,1,1).requires_grad_(True)
# # 0,-5,-0.0039813 for actual poses (not extr)
# # cam_orient_movingcam = p3d_rt.matrix_to_rotation_6d(p3d_rt.axis_angle_to_matrix(torch.tensor([3.14/2,0,0]).float())).repeat(batch_size,num_cams,seq_len,1).requires_grad_(True)
# cam_position = []
# cam_orient = []
# for c in range(len(config["cams_used"])):
#     if c in config["position_changing_cams"]:
#         cam_position.append(torch.tensor([0,0,5]).float().repeat(batch_size,1,seq_len,1).requires_grad_(True).to(device))
#     else:
#         cam_position.append(torch.tensor([0,0,5]).float().repeat(batch_size,1,1,1).requires_grad_(True).to(device))
#     if c in config["orient_changing_cams"]:
#         cam_orient.append(p3d_rt.matrix_to_rotation_6d(p3d_rt.axis_angle_to_matrix(torch.tensor([3.14/2,
#                     0,0]).float().to(device))).repeat(batch_size,1,seq_len,1).requires_grad_(True))
#     else:
#         cam_orient.append(p3d_rt.matrix_to_rotation_6d(p3d_rt.axis_angle_to_matrix(torch.tensor([3.14/2,
#                     0,0]).float().to(device))).repeat(batch_size,1,1,1).requires_grad_(True))
# smpl_trans = torch.zeros(seq_len,3).requires_grad_(True).to(device)
# smpl_orient = p3d_rt.matrix_to_rotation_6d(p3d_rt.axis_angle_to_matrix(torch.zeros(seq_len,3))).requires_grad_(True).to(device)
# smpl_art_motion = p3d_rt.matrix_to_rotation_6d(p3d_rt.axis_angle_to_matrix(torch.zeros(seq_len,21,3))).requires_grad_(True).to(device)
# smpl_shape = torch.zeros(10).unsqueeze(0).requires_grad_(True).to(device)

# # dump yaml file
# yaml.safe_dump(config,open("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/config.yml".format(config["trial_name"],seq_no),"w"))

# # with trange(451,ds.data_lengths[seq_no]-50,50) as seq_t:
# # with torch.autograd.set_detect_anomaly(True):
# with trange(big_seq_start,big_seq_end,fps_scl*(seq_len-overlap)) as seq_t:
#     for seq_start in seq_t:

#         # tensorboard summarywriter
#         os.makedirs("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/{:04d}".format(config["trial_name"],seq_no,seq_start),exist_ok=True)
#         writer = SummaryWriter(log_dir="/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/{:04d}".format(config["trial_name"],seq_no,seq_start), 
#                         filename_suffix="{:04d}".format(seq_start))

#         if dset.lower() == "h36m":
#             # get batch
#             batch = ds.__getitem__(seq_no,seq_start)
#         else:
#             batch = ds.__getitem__(seq_start)
#         j2ds = batch["j2d"].float().unsqueeze(0).to(device)
#         full_j2d.append(j2ds.detach().cpu().numpy())
#         batch_size = j2ds.shape[0]
#         num_cams = j2ds.shape[1]
#         seq_len = j2ds.shape[2]

#         # camera intrinsics
#         if config["optimize_intr"]:
#             focal_len = torch.tensor([3000 for _ in range(num_cams)]).float().requires_grad_(True)
#         else:
#             cam_intr = torch.from_numpy(batch["cam_intr"]).float().unsqueeze(0).to(device)

#         with torch.no_grad():
#             # parameter initialization
#             if config["use_pare_init"]:
#                 cam_position = [cam_position[c].detach().requires_grad_(True) for c in range(num_cams)]
#                 smpl_shape = smpl_shape.detach().requires_grad_(True)
#                 # Decode smpl motion using motion vae
#                 mvae_model.eval()
#                 smpl_motion_init = nmg2smpl((mvae_model.decode(torch.zeros(1,1024).to(device))*mvae_std + mvae_mean).reshape(seq_len,22,9),smpl)
#                 smpl_trans_rf = smpl_motion_init[1:,:3].detach().requires_grad_(True)
#                 smpl_trans_ff = smpl_motion_init[:1,:3].detach().requires_grad_(True)
#                 smpl_orient = p3d_rt.axis_angle_to_matrix(smpl_motion_init[:,3:6])
#                 smpl_art_motion_vp_latent_init = vp_model.encode(batch["pare_poses"][:, :21].reshape(25, 63).to(device)).mean.detach()
#                 # first frame init
#                 pare_orient = p3d_rt.axis_angle_to_matrix(batch["pare_orient"].to(device))
#                 cam_orient = p3d_rt.matrix_to_rotation_6d(torch.matmul(pare_orient[:,0],torch.inverse(smpl_orient[0:1]))).unsqueeze(1)
#                 cam_orient = [cam_orient[c].detach().unsqueeze(0).repeat([1,seq_len,1]).unsqueeze(0).requires_grad_(True) if c in config["orient_changing_cams"] 
#                                 else cam_orient[c,0:1].detach().unsqueeze(0).unsqueeze(0).requires_grad_(True) for c in range(num_cams)]
#                 smpl_orient_rf = p3d_rt.matrix_to_rotation_6d(smpl_orient[1:]).detach().requires_grad_(True)
#                 smpl_orient_ff = p3d_rt.matrix_to_rotation_6d(smpl_orient[:1]).detach().requires_grad_(True)
#             else:
#                 cam_position = [cam_position[c].detach().requires_grad_(True) for c in range(num_cams)]
#                 cam_orient = [cam_orient[c].detach().requires_grad_(True) for c in range(num_cams)]
#                 smpl_trans = smpl_trans.detach().requires_grad_(True)
#                 smpl_orient = smpl_orient.detach().requires_grad_(True)
#                 smpl_art_motion_init = smpl_art_motion.detach()
#                 smpl_shape = smpl_shape.detach().requires_grad_(True)

#             # optimize only selected joints
#             # smpl_art_motion = smpl_art_motion_init[:,[0,1,2,3,4,5,6,7,8,12,13,15,16,17,18,19,20]].detach().requires_grad_(True)
#             # smpl_art_motion_nonOpt = smpl_art_motion_init[:,[9,10,11,14]].detach()
#             smpl_art_motion_vp_latent = smpl_art_motion_vp_latent_init.detach().requires_grad_(True)


#         ################# Optimizer #################
#         if config["optimize_intr"]:
#             optim = torch.optim.Adam(cam_orient + cam_position + [focal_len, smpl_motion, smpl_shape],lr=lr)
#         else:
#             optim0 = torch.optim.Adam(cam_orient + cam_position,lr=lr)
#             optim1 = torch.optim.Adam(cam_orient + cam_position + [smpl_trans_rf,smpl_orient_rf],lr=lr)
#             optim2 = torch.optim.Adam(cam_orient + cam_position + [smpl_art_motion_vp_latent, smpl_trans_rf, smpl_orient_rf, smpl_shape],lr=lr)

#         with trange(n_optim_iters) as t:
#             for i in t:
                
#                 if i < config["n_optim_iters_stage0"]:
#                     stage = 0
#                     optim = optim0
#                 elif i > config["n_optim_iters_stage0"] and i < config["n_optim_iters_stage1"]:
#                     stage = 1
#                     optim = optim1
#                 else:
#                     stage = 2
#                     optim = optim2

#                 ################### FWD pass #####################
#                 # smpl_art_motion_interm = torch.cat([smpl_art_motion[:,:9],smpl_art_motion_nonOpt[:,:3],
#                 #                                     smpl_art_motion[:,9:11],smpl_art_motion_nonOpt[:,3:],
#                 #                                     smpl_art_motion[:,11:]],dim=1)
#                 smpl_art_motion_interm = vp_model.decode(smpl_art_motion_vp_latent)["pose_body"]
#                 smpl_trans = torch.cat([smpl_trans_ff,smpl_trans_rf])
#                 smpl_orient = torch.cat([smpl_orient_ff,smpl_orient_rf])
#                 smpl_motion = torch.cat([smpl_trans.unsqueeze(1),p3d_rt.matrix_to_axis_angle(p3d_rt.rotation_6d_to_matrix(smpl_orient)).unsqueeze(1),
#                                         smpl_art_motion_interm],dim=1).reshape(seq_len,69)
#                 # Decode smpl motion using motion vae
#                 mvae_model.eval()
#                 nmg_repr = (smpl2nmg(smpl_motion,smpl).reshape(-1,seq_len,22*9) - mvae_mean)/mvae_std
#                 smpl_motion_latent = mvae_model.encode(nmg_repr)[:,0]

#                 # SMPL fwd pass
#                 smpl_out = smpl.forward(root_orient = smpl_motion[:,3:6],
#                                             pose_body = smpl_motion[:,6:],
#                                             trans = smpl_motion[:,:3],
#                                             betas = smpl_shape.unsqueeze(1).expand(-1,seq_len,-1).reshape(-1,smpl_shape.shape[-1]))
#                 # smpl_out = smpl2.forward(global_orient = p3d_rt.rotation_6d_to_matrix(smpl_orient).unsqueeze(1),
#                 #                             body_pose = p3d_rt.rotation_6d_to_matrix(smpl_art_motion),
#                 #                             transl = smpl_motion[:,:3],
#                 #                             betas = smpl_shape.unsqueeze(1).expand(-1,seq_len,-1).reshape(-1,smpl_shape.shape[-1]),pose2rot=False)

#                 j3ds = smpl_out.Jtr[:,:22,:]

#                 # camera intrinsics
#                 if config["optimize_intr"]:
#                     cam_intr =  torch.stack([torch.stack([torch.cat([focal_len[c].unsqueeze(0),torch.tensor([0,im_res[0][0]/2])]),
#                             torch.cat([torch.tensor([0]),focal_len[c].unsqueeze(0),torch.tensor([im_res[0][1]/2])]),
#                             torch.tensor([0,0,1])]) for c in range(num_cams)]).unsqueeze(0)

#                 # camera extrinsics
#                 cam_orient_expanded = torch.cat([x.repeat(1,1,seq_len,1) if x.shape[2]==1 else x for x in cam_orient],dim=1)
#                 cam_position_expanded = torch.cat([x.repeat(1,1,seq_len,1) if x.shape[2]==1 else x for x in cam_position],dim=1)
#                 cam_ext_temp = torch.cat([p3d_rt.rotation_6d_to_matrix(cam_orient_expanded),cam_position_expanded.unsqueeze(4)],dim=4)
#                 cam_ext = torch.cat([cam_ext_temp,torch.tensor([0,0,0,1]).type_as(cam_ext_temp).repeat(batch_size,num_cams,seq_len,1,1)],dim=3)

#                 # camera extrinsics
#                 cam_poses = torch.inverse(cam_ext)

#                 # camera projection
#                 proj_j3ds = torch.stack([perspective_projection(j3ds,
#                                 cam_ext[:,i,:,:3,:3].reshape(-1,3,3),
#                                 cam_ext[:,i,:,:3,3].reshape(-1,3),
#                                 cam_intr[:,i].unsqueeze(1).expand(-1,seq_len,-1,-1).reshape(-1,3,3)).reshape(batch_size,seq_len,-1,2) for i in range(num_cams)]).permute(1,0,2,3,4)


#                 ####################### Losses #######################

#                 # reprojection loss
#                 idcs = torch.where(j2ds[0,0,0,:22,2]!=0)[0]

#                 if stage == 0:
#                     loss_2d = (j2ds[:,:,:,idcs,2]*(((proj_j3ds[:,:,:,idcs] - j2ds[:,:,:,idcs,:2])**2).sum(dim=4))).sum(dim=1).mean()
#                 # elif stage == 1:
#                 elif False:
#                     loss_2d = (j2ds[:,:,:,idcs,2]*((gmcclure(proj_j3ds[:,:,:,idcs] - j2ds[:,:,:,idcs,:2],config["gmcclure_sigma"])).sum(dim=4))).sum(dim=1).mean()
#                 else:
#                     loss_2d = (j2ds[:,:,:,idcs,2]*(((proj_j3ds[:,:,:,idcs] - j2ds[:,:,:,idcs,:2])**2).sum(dim=4))).sum(dim=1).mean()

#                 # latent space regularization
#                 loss_z = (smpl_motion_latent*smpl_motion_latent).mean()

#                 # smooth camera motions
#                 loss_cams = ((cam_poses[:,:,1:] - cam_poses[:,:,:-1])**2).mean()

#                 # shape regularization loss
#                 loss_betas = (smpl_shape*smpl_shape).mean()

#                 # ground penetration loss
#                 loss_human_gp = torch.nn.functional.relu(-j3ds[:,:,2]).mean()

#                 # camera ground penetration loss
#                 loss_cam_gp = torch.nn.functional.relu(-cam_poses[:,:,:,2,3]).mean()

#                 # loss vposer
#                 loss_vp = (smpl_art_motion_vp_latent*smpl_art_motion_vp_latent).mean()

#                 # loss smpl in front
#                 loss_smpl_in_front = torch.nn.functional.relu(-cam_ext[:,:,:,2,3]).mean()

#                 # total loss
#                 loss = loss_2d_weight[stage] * loss_2d + \
#                         loss_z_weight[stage] * loss_z + \
#                             loss_cams_weight[stage] * loss_cams + \
#                                 loss_betas_weight[stage] * loss_betas + \
#                                     loss_human_gp_weight[stage] * loss_human_gp + \
#                                         loss_cam_gp_weight[stage] * loss_cam_gp + \
#                                             loss_vp_weight[stage] * loss_vp + \
#                                                 loss_smpl_in_front_weight[stage] * loss_smpl_in_front

#                 # Viz
#                 if False:
#                 # if i == 0:
#                     full_images_paths = batch["full_im_paths"]
#                     num_cams = len(full_images_paths)
#                     seq_len = len(full_images_paths[0])
#                     # random index
#                     idx = np.random.randint(cam_ext.shape[0])
#                     rend_ims = [np.zeros([seq_len,im_res[c][1],im_res[c][0],3]) for c in range(num_cams)]
#                     for cam in tqdm(range(num_cams)):
#                         for s in range(seq_len):
#                             im = cv2.imread(full_images_paths[cam][s])[:im_res[cam][1],:im_res[cam][0],::-1]/255.
#                             rend_ims[cam][s] = renderer[cam](smpl_out.v.view(-1,seq_len,6890,3)[idx,s].detach().cpu().numpy(),
#                                                 cam_ext[idx,cam,s,:3,3].detach().cpu().numpy(),
#                                                 cam_ext[idx,cam,s,:3,:3].unsqueeze(0).detach().cpu().numpy(),
#                                                 im,intr=cam_intr[idx,cam].detach().cpu().numpy(),
#                                                 faces=smpl.f.detach().cpu().numpy())
#                             for joint in range(j2ds.shape[3]):
#                                 rend_ims[cam][s] = rend_ims[cam][s]*255
#                                 cv2.circle(rend_ims[cam][s],(int(j2ds[idx,cam,s,joint,0]),
#                                             int(j2ds[idx,cam,s,joint,1])),10,(255,255,255),-1)
#                                 rend_ims[cam][s] = rend_ims[cam][s]/255
                    
#                     for c in range(num_cams):
#                         imageio.mimsave("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/test_{:05d}_cam_{:02d}_itr_{:04d}.gif".format(config["trial_name"],
#                                     seq_no,seq_start,c,i),list(rend_ims[c][:,::2,::2]))
#                         # imageio.mimsave("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/test_{:05d}_cam_{:02d}.gif".format(config["trial_name"],seq_no,seq_start,c),
#                         #             [make_grid(torch.from_numpy(rend_ims[:,i]).permute(0,3,1,2),
#                         #                 nrow=rend_ims.shape[0]).permute(1,2,0).cpu().numpy()[::5,::5] for i in range(rend_ims.shape[1])])


#                     np.savez("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/test_{:05d}_itr_{:04d}".format(config["trial_name"],seq_no,seq_start,i),
#                         verts=smpl_out.v.detach().cpu().numpy(),
#                         cam_trans=cam_poses[:,:,:,:3,3].detach().cpu().numpy(),
#                         cam_rots=p3d_rt.matrix_to_quaternion(cam_poses[:,:,:,:3,:3]).detach().cpu().numpy())


#                 # zero grad optimizer
#                 optim.zero_grad()
                
#                 # backward
#                 loss.backward()

#                 # optim step
#                 optim.step()

#                 # loss dict
#                 loss_dict = {"loss":loss.item(),
#                                 "loss_2d":loss_2d.item(),
#                                 "loss_z":loss_z.item(),
#                                 "loss_cams":loss_cams.item(),
#                                 "loss_betas":loss_betas.item(),
#                                 "loss_human_gp":loss_human_gp.item(),
#                                 "loss_cam_gp":loss_cam_gp.item(),
#                                 "loss_vp":loss_vp.item(),
#                                 "loss_smpl_front":loss_smpl_in_front.item()}

#                 # print loss
#                 t.set_postfix(loss_dict)

#                 # track losses
#                 for k,v in loss_dict.items():
#                     writer.add_scalar(k,v,global_step=i)

#                 # Viz
#                 # if i == n_optim_iters-1 or i == config["n_optim_iters_stage1"]-1 or i == config["n_optim_iters_stage0"]-1:
#                 # if i == n_optim_iters-1:
#                 if False:
#                     full_images_paths = batch["full_im_paths"]
#                     num_cams = len(full_images_paths)
#                     seq_len = len(full_images_paths[0])
#                     # random index
#                     idx = np.random.randint(cam_ext.shape[0])
#                     rend_ims = [np.zeros([seq_len,np.ceil(im_res[c][1]/viz_dwnsample).astype(int),np.ceil(im_res[c][0]/viz_dwnsample).astype(int),3]) for c in range(num_cams)]
#                     for cam in tqdm(range(num_cams)):
#                         for s in range(seq_len):
#                             im = cv2.imread(full_images_paths[cam][s])[:im_res[cam][1]:viz_dwnsample,:im_res[cam][0]:viz_dwnsample,::-1]/255.
#                             temp_intr = cam_intr[idx,cam].detach().cpu().numpy()
#                             temp_intr[:2,:3] = temp_intr[:2,:3] / viz_dwnsample
#                             temp_v = smpl_out.v.view(-1,seq_len,6890,3)[idx,s].detach().cpu().numpy()
#                             temp_v[:,2] = temp_v[:,2]*viz_dwnsample
#                             rend_ims[cam][s] = renderer[cam](temp_v,
#                                                 cam_ext[idx,cam,s,:3,3].detach().cpu().numpy(),
#                                                 cam_ext[idx,cam,s,:3,:3].unsqueeze(0).detach().cpu().numpy(),
#                                                 im,intr=temp_intr,
#                                                 faces=smpl.f.detach().cpu().numpy())
#                             for joint in range(j2ds.shape[3]):
#                                 rend_ims[cam][s] = rend_ims[cam][s]*255
#                                 cv2.circle(rend_ims[cam][s],(int(j2ds[idx,cam,s,joint,0]/viz_dwnsample),
#                                             int(j2ds[idx,cam,s,joint,1]/viz_dwnsample)),10,(255,255,255),-1)
#                                 rend_ims[cam][s] = rend_ims[cam][s]/255
#                     for c in range(num_cams):
#                         imageio.mimsave("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/test_{:05d}_cam_{:02d}_itr_{:04d}.gif".format(config["trial_name"],
#                                     seq_no,seq_start,c,i),list(rend_ims[c][:,::2,::2]))
#                         # imageio.mimsave("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/test_{:05d}_cam_{:02d}.gif".format(config["trial_name"],seq_no,seq_start,c),
#                         #             [make_grid(torch.from_numpy(rend_ims[:,i]).permute(0,3,1,2),
#                         #                 nrow=rend_ims.shape[0]).permute(1,2,0).cpu().numpy()[::5,::5] for i in range(rend_ims.shape[1])])


#                     np.savez("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/test_{:05d}_itr_{:04d}".format(config["trial_name"],seq_no,seq_start,i),
#                         verts=smpl_out.v.detach().cpu().numpy(),
#                         cam_trans=cam_poses[:,:,:,:3,3].detach().cpu().numpy(),
#                         cam_rots=p3d_rt.matrix_to_quaternion(cam_poses[:,:,:,:3,:3]).detach().cpu().numpy())

#         with torch.no_grad():
#             if len(full_cam_orient) == 0:
#                 full_cam_orient.append(p3d_rt.matrix_to_quaternion(cam_poses[:,:,:,:3,:3]).detach().cpu().numpy())
#                 full_cam_position.append(cam_poses[:,:,:,:3,3].detach().cpu().numpy())
#                 full_smpl_verts.append(smpl_out.v.detach().cpu().numpy())
#                 full_smpl_shape.append(smpl_shape.detach().cpu().numpy())
#                 full_smpl_motion_latent.append(smpl_motion_latent.detach())
#                 full_vp_latent.append(smpl_art_motion_vp_latent.detach().cpu().numpy())
#                 full_cam_ext.append(cam_ext.detach().cpu().numpy())
#                 full_smpl_orient.append(p3d_rt.matrix_to_axis_angle(p3d_rt.rotation_6d_to_matrix(smpl_orient)).detach().cpu().numpy())
#                 full_smpl_trans.append(smpl_trans.detach().cpu().numpy())
#                 full_smpl_motion.append(smpl_motion[:,6:].detach().cpu().numpy())
#                 # save last smpl orient
#                 last_smpl_orient = smpl_motion[:,3:6].detach()
#                 last_smpl_trans = smpl_motion[:,:3].detach()
#             else:
#                 pos, ori = geometry.get_ground_point(last_smpl_trans[-overlap+1],p3d_rt.axis_angle_to_matrix(last_smpl_orient[-overlap+1]))
#                 prev_tfm = to_homogeneous(ori,pos)
#                 curr_pos, curr_ori = geometry.get_ground_point(smpl_motion[0,:3],p3d_rt.axis_angle_to_matrix(smpl_motion[0,3:6]))
#                 curr_tfm = to_homogeneous(curr_ori,curr_pos)
#                 tfm = torch.matmul(prev_tfm,torch.inverse(curr_tfm))
#                 curr_smpl = to_homogeneous(p3d_rt.axis_angle_to_matrix(smpl_motion[:,3:6]),smpl_motion[:,:3])
#                 tfmd_smpl = torch.matmul(tfm,curr_smpl)
#                 updated_smpl_orient = p3d_rt.matrix_to_axis_angle(tfmd_smpl[:,:3,:3])
#                 updated_smpl_trans = tfmd_smpl[:,:3,3]

#                 smpl_out_updated = smpl.forward(root_orient = updated_smpl_orient,
#                                             pose_body = smpl_motion[:,6:],
#                                             trans = updated_smpl_trans,
#                                             betas = smpl_shape.unsqueeze(1).expand(-1,seq_len,-1).reshape(-1,smpl_shape.shape[-1]))
#                 updated_cam_orient = torch.matmul(tfm[:,:3,:3],cam_poses[0,:,:,:3,:3])
#                 updated_cam_position = torch.matmul(tfm[:,:3,:3],cam_poses[0,:,:,:3,3:]).squeeze(3) + tfm[0,:3,3]
#                 full_cam_orient.append(p3d_rt.matrix_to_quaternion(updated_cam_orient).unsqueeze(0).detach().cpu().numpy())
#                 full_cam_position.append(updated_cam_position.unsqueeze(0).detach().cpu().numpy())
#                 full_smpl_verts.append(smpl_out_updated.v.detach().cpu().numpy())
#                 full_smpl_shape.append(smpl_shape.detach().cpu().numpy())
#                 full_smpl_motion_latent.append(smpl_motion_latent.detach())
#                 full_vp_latent.append(smpl_art_motion_vp_latent.detach().cpu().numpy())
#                 full_cam_ext.append(cam_ext.detach().cpu().numpy())
#                 full_smpl_orient.append(updated_smpl_orient.detach().cpu().numpy())
#                 full_smpl_trans.append(updated_smpl_trans.detach().cpu().numpy())
#                 full_smpl_motion.append(smpl_motion[:,6:].detach().cpu().numpy())

#                 # save last smpl orient
#                 last_smpl_orient = p3d_rt.matrix_to_axis_angle(tfmd_smpl[:,:3,:3]).detach()
#                 last_smpl_trans = tfmd_smpl[:,:3,3].detach()


# full_cam_orient = np.concatenate(full_cam_orient,axis=2)
# full_cam_position = np.concatenate(full_cam_position,axis=2)
# full_smpl_verts = np.concatenate(full_smpl_verts)
# full_smpl_shape = np.concatenate(full_smpl_shape)
# full_smpl_motion_latent = torch.cat(full_smpl_motion_latent,dim=0)
# full_cam_ext = np.concatenate(full_cam_ext,axis=2)
# full_smpl_orient = np.concatenate(full_smpl_orient,axis=0)
# full_smpl_trans = np.concatenate(full_smpl_trans,axis=0)
# full_smpl_motion = np.concatenate(full_smpl_motion,axis=0)
# full_vp_latent = np.concatenate(full_vp_latent,axis=0)
# full_j2d = np.concatenate(full_j2d,axis=2)

# # imageio.mimsave("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/test/test.gif",overlay_gifs,fps=30)

# np.savez("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/test_full".format(config["trial_name"],seq_no),
#                 verts=full_smpl_verts,
#                 cam_trans=full_cam_position,
#                 cam_rots=full_cam_orient)


# non_overlap_cam_position = np.concatenate([full_cam_position[:,:,i:i+seq_len-overlap] for i in range(0,full_cam_position.shape[2]-seq_len,seq_len)] + 
#                             [full_cam_position[:,:,-seq_len:]],axis=2)
# non_overlap_cam_orient = np.concatenate([full_cam_orient[:,:,i:i+seq_len-overlap] for i in range(0,full_cam_orient.shape[2]-seq_len,seq_len)] +
#                             [full_cam_orient[:,:,-seq_len:]],axis=2)
# non_overlap_smpl_trans = np.concatenate([full_smpl_trans[i:i+seq_len-overlap] for i in range(0,full_smpl_trans.shape[0]-seq_len,seq_len)] +
#                             [full_smpl_trans[-seq_len:]],axis=0)
# non_overlap_smpl_orient = np.concatenate([full_smpl_orient[i:i+seq_len-overlap] for i in range(0,full_smpl_orient.shape[0]-seq_len,seq_len)] +
#                             [full_smpl_orient[-seq_len:]],axis=0)
# non_overlap_smpl_motion = np.concatenate([full_smpl_motion[i:i+seq_len-overlap] for i in range(0,full_smpl_motion.shape[0]-seq_len,seq_len)] +
#                             [full_smpl_motion[-seq_len:]],axis=0)
# non_overlap_vp_latent = np.concatenate([full_vp_latent[i:i+seq_len-overlap] for i in range(0,full_vp_latent.shape[0]-seq_len,seq_len)] +
#                             [full_vp_latent[-seq_len:]],axis=0)
# non_overlap_j2ds = np.concatenate([full_j2d[:,:,i:i+seq_len-overlap] for i in range(0,full_j2d.shape[2]-seq_len,seq_len)] +
#                             [full_j2d[:,:,-seq_len:]],axis=2)
# non_overlap_smpl_verts = np.concatenate([full_smpl_verts[i:i+seq_len-overlap] for i in range(0,full_smpl_verts.shape[0]-seq_len,seq_len)] +
#                             [full_smpl_verts[-seq_len:]],axis=0)
# non_overlap_cam_ext = np.concatenate([full_cam_ext[:,:,i:i+seq_len-overlap] for i in range(0,full_cam_ext.shape[2]-seq_len,seq_len)] +
#                             [full_cam_ext[:,:,-seq_len:]],axis=2)

# np.savez("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/test_full_non_overlap".format(config["trial_name"],seq_no),
#                 verts=non_overlap_smpl_verts,
#                 cam_trans=non_overlap_cam_position,
#                 cam_rots=non_overlap_cam_orient)




# ###############################################################################################################################

# cam_ext = torch.from_numpy(non_overlap_cam_ext).float().to(device)

# seq_len = non_overlap_vp_latent.shape[0]

# # Full fittings
# full_cam_orient = []
# full_cam_position = []
# full_smpl_verts = []
# full_smpl_shape = []
# full_smpl_motion_latent = []
# full_cam_ext = []
# full_smpl_orient = []
# full_smpl_trans = []
# full_smpl_motion = []
# # overlay_gifs = []

# cam_ext_position = []
# cam_ext_orient = []
# for c in range(len(config["cams_used"])):
#     if c in config["position_changing_cams"]:
#         cam_ext_position.append(cam_ext[:,c:c+1,:,:3,3].requires_grad_(True).to(device))
#     else:
#         cam_ext_position.append(cam_ext[:,c:c+1,0:1,:3,3].requires_grad_(True).to(device))
#     if c in config["orient_changing_cams"]:
#         cam_ext_orient.append(p3d_rt.matrix_to_rotation_6d(cam_ext[:,c:c+1,:,:3,:3]).requires_grad_(True).to(device))
#     else:
#         cam_ext_orient.append(p3d_rt.matrix_to_rotation_6d(cam_ext[:,c:c+1,0:1,:3,:3]).detach().requires_grad_(True).to(device))
# smpl_trans = torch.from_numpy(non_overlap_smpl_trans).to(device)
# smpl_orient = p3d_rt.matrix_to_rotation_6d(p3d_rt.axis_angle_to_matrix(torch.from_numpy(non_overlap_smpl_orient).to(device)))
# smpl_art_motion_vp_latent = torch.from_numpy(non_overlap_vp_latent).float().detach().to(device).requires_grad_(True)
# smpl_shape = torch.zeros(10).unsqueeze(0).to(device).requires_grad_(True)

# # tensorboard summarywriter
# os.makedirs("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/fullopt".format(config["trial_name"],seq_no),exist_ok=True)
# writer = SummaryWriter(log_dir="/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/fullopt".format(config["trial_name"],seq_no), 
#                 filename_suffix="{:04d}".format(seq_start))


# j2ds = torch.from_numpy(non_overlap_j2ds).float().to(device)

# with torch.no_grad():
# # parameter initialization
#     # Decode smpl motion using motion vae
#     smpl_trans_rf = smpl_trans[1:].detach().requires_grad_(True)
#     smpl_trans_ff = smpl_trans[:1].detach()
#     smpl_orient_rf = smpl_orient[1:].detach().requires_grad_(True)
#     smpl_orient_ff = smpl_orient[:1].detach()


# ################# Optimizer #################
# if config["optimize_intr"]:
#     optim = torch.optim.Adam(cam_ext_orient + cam_ext_position + [focal_len, smpl_motion, smpl_shape],lr=lr)
# else:
#     optim0 = torch.optim.Adam(cam_orient + cam_position,lr=lr)
#     optim1 = torch.optim.Adam(cam_orient + cam_position,lr=lr)
#     optim2 = torch.optim.Adam(cam_orient + cam_position + [smpl_art_motion_vp_latent, smpl_trans_rf, smpl_orient_rf, smpl_shape],lr=lr)

# with trange(n_optim_iters) as t:
#     for i in t:

#         if i < config["n_optim_iters_stage0"]:
#             stage = 0
#             optim = optim0
#         elif i > config["n_optim_iters_stage0"] and i < config["n_optim_iters_stage1"]:
#             stage = 1
#             optim = optim1
#         else:
#             stage = 2
#             optim = optim2

#         ################### FWD pass #####################
#         smpl_art_motion_interm = vp_model.decode(smpl_art_motion_vp_latent)["pose_body"]
#         smpl_trans = torch.cat([smpl_trans_ff,smpl_trans_rf])
#         smpl_orient = torch.cat([smpl_orient_ff,smpl_orient_rf])
#         smpl_motion = torch.cat([smpl_trans.unsqueeze(1),p3d_rt.matrix_to_axis_angle(p3d_rt.rotation_6d_to_matrix(smpl_orient)).unsqueeze(1),
#                                 smpl_art_motion_interm],dim=1).reshape(seq_len,69)
        
        
#         # encode smpl motion using motion vae
#         mvae_model.eval()
#         smpl_motion_latent = []
#         for strt_idx in range(0,seq_len-overlap,25-overlap):
#             curr_pos, curr_ori = geometry.get_ground_point(smpl_trans[strt_idx],p3d_rt.rotation_6d_to_matrix(smpl_orient[strt_idx]))
#             curr_tfm = to_homogeneous(curr_ori,curr_pos)
#             curr_root_pose = to_homogeneous(p3d_rt.rotation_6d_to_matrix(smpl_orient[strt_idx:strt_idx+25]),smpl_trans[strt_idx:strt_idx+25])
#             canonical_root_pose = torch.matmul(torch.inverse(curr_tfm),curr_root_pose)
#             canonical_smpl_motion = torch.cat([canonical_root_pose[:,:3,3],
#                                     p3d_rt.matrix_to_axis_angle(canonical_root_pose[:,:3,:3]),
#                                     smpl_motion[strt_idx:strt_idx+25,6:]],dim=1)
#             nmg_repr = (smpl2nmg(canonical_smpl_motion,smpl).reshape(-1,25,22*9) - mvae_mean)/mvae_std
#             smpl_motion_latent.append(mvae_model.encode(nmg_repr)[:,0])

#         # SMPL fwd pass
#         smpl_out = smpl.forward(root_orient = smpl_motion[:,3:6],
#                                     pose_body = smpl_motion[:,6:],
#                                     trans = smpl_motion[:,:3],
#                                     betas = smpl_shape.unsqueeze(1).expand(-1,seq_len,-1).reshape(-1,smpl_shape.shape[-1]))

#         j3ds = smpl_out.Jtr[:,:22,:]

#         # camera extrinsics
#         cam_ext_orient_expanded = torch.cat([x.repeat(1,1,seq_len,1) if x.shape[2]==1 else x for x in cam_ext_orient],dim=1)
#         cam_ext_position_expanded = torch.cat([x.repeat(1,1,seq_len,1) if x.shape[2]==1 else x for x in cam_ext_position],dim=1)
#         cam_ext_temp = torch.cat([p3d_rt.rotation_6d_to_matrix(cam_ext_orient_expanded),cam_ext_position_expanded.unsqueeze(4)],dim=4)
#         cam_ext = torch.cat([cam_ext_temp,torch.tensor([0,0,0,1]).type_as(cam_ext_temp).repeat(batch_size,num_cams,seq_len,1,1)],dim=3)

#         # camera extrinsics
#         cam_poses = torch.inverse(cam_ext)

#         # camera projection
#         proj_j3ds = torch.stack([perspective_projection(j3ds,
#                         cam_ext[:,i,:,:3,:3].reshape(-1,3,3),
#                         cam_ext[:,i,:,:3,3].reshape(-1,3),
#                         cam_intr[:,i].unsqueeze(1).expand(-1,seq_len,-1,-1).reshape(-1,3,3)).reshape(batch_size,seq_len,-1,2) for i in range(num_cams)]).permute(1,0,2,3,4)


#         ####################### Losses #######################

#         # reprojection loss
#         # if stage == 1:
#         if False:
#             loss_2d = (j2ds[:,:,:,:22,2]*((gmcclure(proj_j3ds[:,:,:,:22] - j2ds[:,:,:,:22,:2],config["gmcclure_sigma"])).sum(dim=4))).sum(dim=1).mean()
#         else:
#             loss_2d = (j2ds[:,:,:,:22,2]*(((proj_j3ds[:,:,:,:22] - j2ds[:,:,:,:22,:2])**2).sum(dim=4))).sum(dim=1).mean()

#         # latent space regularization
#         loss_z = torch.cat([(x*x).mean().unsqueeze(0) for x in smpl_motion_latent]).sum()

#         # smooth camera motions
#         loss_cams = ((cam_poses[:,:,1:] - cam_poses[:,:,:-1])**2).mean()

#         # shape regularization loss
#         loss_betas = (smpl_shape*smpl_shape).mean()

#         # ground penetration loss
#         loss_human_gp = torch.nn.functional.relu(-j3ds[:,:,2]).mean()

#         # camera ground penetration loss
#         loss_cam_gp = torch.nn.functional.relu(-cam_poses[:,:,:,2,3]).mean()

#         # loss vposer
#         loss_vp = (smpl_art_motion_vp_latent*smpl_art_motion_vp_latent).mean()

#         # loss smpl in front
#         loss_smpl_in_front = torch.nn.functional.relu(-cam_ext[:,:,:,2,3]).mean()

#         # total loss
#         loss = loss_2d_weight[stage] * loss_2d + \
#                 loss_z_weight[stage] * loss_z + \
#                     loss_cams_weight[stage] * loss_cams + \
#                         loss_betas_weight[stage] * loss_betas + \
#                             loss_human_gp_weight[stage] * loss_human_gp + \
#                                 loss_cam_gp_weight[stage] * loss_cam_gp + \
#                                     loss_vp_weight[stage] * loss_vp + \
#                                         loss_smpl_in_front_weight[stage] * loss_smpl_in_front

#         # Viz
#         if False:
#         # if i == 0:
#             full_images_paths = batch["full_im_paths"]
#             num_cams = len(full_images_paths)
#             seq_len = len(full_images_paths[0])
#             # random index
#             idx = np.random.randint(cam_ext.shape[0])
#             rend_ims = [np.zeros([seq_len,im_res[c][1],im_res[c][0],3]) for c in range(num_cams)]
#             for cam in tqdm(range(num_cams)):
#                 for s in range(seq_len):
#                     im = cv2.imread(full_images_paths[cam][s])[:im_res[cam][1],:im_res[cam][0],::-1]/255.
#                     rend_ims[cam][s] = renderer[cam](smpl_out.v.view(-1,seq_len,6890,3)[idx,s].detach().cpu().numpy(),
#                                         cam_ext[idx,cam,s,:3,3].detach().cpu().numpy(),
#                                         cam_ext[idx,cam,s,:3,:3].unsqueeze(0).detach().cpu().numpy(),
#                                         im,intr=cam_intr[idx,cam].detach().cpu().numpy(),
#                                         faces=smpl.f.detach().cpu().numpy())
#                     for joint in range(j2ds.shape[3]):
#                         rend_ims[cam][s] = rend_ims[cam][s]*255
#                         cv2.circle(rend_ims[cam][s],(int(j2ds[idx,cam,s,joint,0]),
#                                     int(j2ds[idx,cam,s,joint,1])),10,(255,255,255),-1)
#                         rend_ims[cam][s] = rend_ims[cam][s]/255
            
#             for c in range(num_cams):
#                 imageio.mimsave("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/test_{:05d}_cam_{:02d}_itr_{:04d}.gif".format(config["trial_name"],
#                             seq_no,seq_start,c,i),list(rend_ims[c][:,::2,::2]))
#                 # imageio.mimsave("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/test_{:05d}_cam_{:02d}.gif".format(config["trial_name"],seq_no,seq_start,c),
#                 #             [make_grid(torch.from_numpy(rend_ims[:,i]).permute(0,3,1,2),
#                 #                 nrow=rend_ims.shape[0]).permute(1,2,0).cpu().numpy()[::5,::5] for i in range(rend_ims.shape[1])])


#             np.savez("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/test_{:05d}_itr_{:04d}".format(config["trial_name"],seq_no,seq_start,i),
#                 verts=smpl_out.v.detach().cpu().numpy(),
#                 cam_trans=cam_poses[:,:,:,:3,3].detach().cpu().numpy(),
#                 cam_rots=p3d_rt.matrix_to_quaternion(cam_poses[:,:,:,:3,:3]).detach().cpu().numpy())


#         # zero grad optimizer
#         optim.zero_grad()
        
#         # backward
#         loss.backward()

#         # optim step
#         optim.step()

#         # loss dict
#         loss_dict = {"loss":loss.item(),
#                         "loss_2d":loss_2d.item(),
#                         "loss_z":loss_z.item(),
#                         "loss_cams":loss_cams.item(),
#                         "loss_betas":loss_betas.item(),
#                         "loss_human_gp":loss_human_gp.item(),
#                         "loss_cam_gp":loss_cam_gp.item(),
#                         "loss_vp":loss_vp.item(),
#                         "loss_smpl_front":loss_smpl_in_front.item()}

#         # print loss
#         t.set_postfix(loss_dict)

#         # track losses
#         for k,v in loss_dict.items():
#             writer.add_scalar(k,v,global_step=i)

#         # Viz
#         if False:
#         # if i == n_optim_iters-1 or i == config["n_optim_iters_stage1"]-1:
#             full_images_paths = batch["full_im_paths"]
#             num_cams = len(full_images_paths)
#             seq_len = len(full_images_paths[0])
#             # random index
#             idx = np.random.randint(cam_ext.shape[0])
#             rend_ims = [np.zeros([seq_len,im_res[c][1],im_res[c][0],3]) for c in range(num_cams)]
#             for cam in tqdm(range(num_cams)):
#                 for s in range(seq_len):
#                     im = cv2.imread(full_images_paths[cam][s])[:im_res[cam][1],:im_res[cam][0],::-1]/255.
#                     rend_ims[cam][s] = renderer[cam](smpl_out.v.view(-1,seq_len,6890,3)[idx,s].detach().cpu().numpy(),
#                                         cam_ext[idx,cam,s,:3,3].detach().cpu().numpy(),
#                                         cam_ext[idx,cam,s,:3,:3].unsqueeze(0).detach().cpu().numpy(),
#                                         im,intr=cam_intr[idx,cam].detach().cpu().numpy(),
#                                         faces=smpl.f.detach().cpu().numpy())
#                     for joint in range(j2ds.shape[3]):
#                         rend_ims[cam][s] = rend_ims[cam][s]*255
#                         cv2.circle(rend_ims[cam][s],(int(j2ds[idx,cam,s,joint,0]),
#                                     int(j2ds[idx,cam,s,joint,1])),10,(255,255,255),-1)
#                         rend_ims[cam][s] = rend_ims[cam][s]/255
#             for c in range(num_cams):
#                 imageio.mimsave("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/test_{:05d}_cam_{:02d}_itr_{:04d}.gif".format(config["trial_name"],
#                             seq_no,seq_start,c,i),list(rend_ims[c][:,::2,::2]))
#                 # imageio.mimsave("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/test_{:05d}_cam_{:02d}.gif".format(config["trial_name"],seq_no,seq_start,c),
#                 #             [make_grid(torch.from_numpy(rend_ims[:,i]).permute(0,3,1,2),
#                 #                 nrow=rend_ims.shape[0]).permute(1,2,0).cpu().numpy()[::5,::5] for i in range(rend_ims.shape[1])])


#             np.savez("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/test_{:05d}_itr_{:04d}".format(config["trial_name"],seq_no,seq_start,i),
#                 verts=smpl_out.v.detach().cpu().numpy(),
#                 cam_trans=cam_poses[:,:,:,:3,3].detach().cpu().numpy(),
#                 cam_rots=p3d_rt.matrix_to_quaternion(cam_poses[:,:,:,:3,:3]).detach().cpu().numpy())

# with torch.no_grad():
#     final_cam_orient = p3d_rt.matrix_to_quaternion(cam_poses[:,:,:,:3,:3]).detach().cpu().numpy()
#     final_cam_position = cam_poses[:,:,:,:3,3].detach().cpu().numpy()
#     final_smpl_verts = smpl_out.v.detach().cpu().numpy()
#     final_smpl_shape = smpl_shape.detach().cpu().numpy()
#     final_smpl_motion = smpl_motion.detach().cpu().numpy()
#     final_cam_ext = cam_ext.detach()

# np.savez("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/{}/{:04d}/test_final".format(config["trial_name"],seq_no),
#                 verts=final_smpl_verts,
#                 shape=final_smpl_shape,
#                 motion=final_smpl_motion,
#                 cam_trans=final_cam_position,
#                 cam_rots=final_cam_orient)


# # %% Evaluation
