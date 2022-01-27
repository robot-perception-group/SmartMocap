import torch
import os
from os.path import join as ospj
import pickle as pkl
from torch.utils.data import Dataset
import numpy as np
import sys
import glob
import numpy as np
import h5py
import json
from tqdm import tqdm
import cv2
from torchvision import transforms

from ..utils.utils import resize_with_pad

# remove nose as head
op_map2smpl = np.array([8,12,9,-1,13,10,-1,14,11,-1,19,22,1,-1,-1,-1,5,2,6,3,7,4,-1,-1])
al_map2smpl = np.array([-1,11,8,-1,12,9,-1,13,10,-1,-1,-1,1,-1,-1,-1,5,2,6,3,7,4,-1,-1])
# smpl_map2op = np.array([])

class h36m(Dataset):
    def __init__(self,hparams,subjects=["S1","S5","S6","S7","S8"]):
        super().__init__()
        
        self.seq_dirs = {"c0":[],"c1":[],"c2":[],"c3":[]}
        # get sequences list
        for sub in tqdm(subjects):
            self.seq_dirs["c0"] += glob.glob(ospj(hparams["data_datapath"],"images",sub,"*.54138969"))
            self.seq_dirs["c1"] += glob.glob(ospj(hparams["data_datapath"],"images",sub,"*.55011271"))
            self.seq_dirs["c2"] += glob.glob(ospj(hparams["data_datapath"],"images",sub,"*.58860488"))
            self.seq_dirs["c3"] += glob.glob(ospj(hparams["data_datapath"],"images",sub,"*.60457274"))
        
        # sort
        self.seq_dirs["c0"] = sorted(self.seq_dirs["c0"])
        self.seq_dirs["c1"] = sorted(self.seq_dirs["c1"])
        self.seq_dirs["c2"] = sorted(self.seq_dirs["c2"])
        self.seq_dirs["c3"] = sorted(self.seq_dirs["c3"])

        # data lengths
        dl0 = np.array([len(glob.glob(ospj(x,"*.jpg"))) for x in tqdm(self.seq_dirs["c0"])])
        dl1 = np.array([len(glob.glob(ospj(x,"*.jpg"))) for x in tqdm(self.seq_dirs["c1"])])
        dl2 = np.array([len(glob.glob(ospj(x,"*.jpg"))) for x in tqdm(self.seq_dirs["c2"])])
        dl3 = np.array([len(glob.glob(ospj(x,"*.jpg"))) for x in tqdm(self.seq_dirs["c3"])])
        data_lengths = np.stack([dl0,dl1,dl2,dl3])
        self.data_lengths = np.min(data_lengths,axis=0)

        # sequence length
        self.seq_len = 2 * hparams["data_seq_len"]

        # get camera parameters
        self.cam_extr = {"c0":{},"c1":{},"c2":{},"c3":{}}
        self.cam_intr = {"c0":{},"c1":{},"c2":{},"c3":{}}
        with h5py.File(ospj(hparams["data_datapath"],"h36m_cameras.h5"),'r') as cam_db:
            for sub in subjects:
                self.cam_extr["c0"][sub] = np.concatenate([cam_db['subject'+sub[1]]['camera1']['R'][()],
                                    cam_db['subject'+sub[1]]['camera1']['T'][()]],axis=1)
                self.cam_extr["c1"][sub] = np.concatenate([cam_db['subject'+sub[1]]['camera2']['R'][()],
                                    cam_db['subject'+sub[1]]['camera2']['T'][()]],axis=1)
                self.cam_extr["c2"][sub] = np.concatenate([cam_db['subject'+sub[1]]['camera1']['R'][()],
                                    cam_db['subject'+sub[1]]['camera1']['T'][()]],axis=1)
                self.cam_extr["c3"][sub] = np.concatenate([cam_db['subject'+sub[1]]['camera2']['R'][()],
                                    cam_db['subject'+sub[1]]['camera2']['T'][()]],axis=1)
                
                intr0 = np.eye(3)
                intr0[0,0],intr0[1,1] = cam_db['subject'+sub[1]]['camera1']['f'][()][:,0]
                intr0[:2,2] = cam_db['subject'+sub[1]]['camera1']['c'][()][:,0]
                self.cam_intr["c0"][sub] = intr0
                intr1 = np.eye(3)
                intr1[0,0],intr1[1,1] = cam_db['subject'+sub[1]]['camera2']['f'][()][:,0]
                intr1[:2,2] = cam_db['subject'+sub[1]]['camera2']['c'][()][:,0]
                self.cam_intr["c1"][sub] = intr1
                intr2 = np.eye(3)
                intr2[0,0],intr2[1,1] = cam_db['subject'+sub[1]]['camera1']['f'][()][:,0]
                intr2[:2,2] = cam_db['subject'+sub[1]]['camera1']['c'][()][:,0]
                self.cam_intr["c2"][sub] = intr2
                intr3 = np.eye(3)
                intr3[0,0],intr3[1,1] = cam_db['subject'+sub[1]]['camera2']['f'][()][:,0]
                intr3[:2,2] = cam_db['subject'+sub[1]]['camera2']['c'][()][:,0]
                self.cam_intr["c3"][sub] = intr3
        
        
        # openpose and Alphapose results
        self.opose_res = {"c0":[],"c1":[],"c2":[],"c3":[]}
        self.apose_res = {"c0":[],"c1":[],"c2":[],"c3":[]}
        for cam in tqdm(["c0","c1","c2","c3"]):
            for seq_dir in tqdm(self.seq_dirs[cam]):
                op_fl = seq_dir.split("/")
                op_fl[-3] = "opose_res"
                op_fl = "/".join(op_fl)
                opose_raw = pkl.load(open(op_fl+".pkl","rb"))
                ap_fl = seq_dir.split("/")
                ap_fl[-3] = "apose_res"
                ap_fl = "/".join(ap_fl) + "/alphapose-results.json"
                apose_raw = json.load(open(ap_fl,"r"))
                opose = np.zeros([int(sorted(opose_raw.keys())[-1]),24,3])
                apose = np.zeros([int(sorted(apose_raw.keys())[-1].split(".")[0]),24,3])
                for i in range(len(opose_raw.keys())):
                    try:
                        opose[i] = opose_raw["{:06d}".format(i+1)]["pose"][0,op_map2smpl]
                        opose[i][op_map2smpl==-1,:] = 0
                    except:
                        pass
                    try:
                        apose[i] = np.reshape(apose_raw["{:06d}".format(i+1)+".jpg"]["people"][0]["pose_keypoints_2d"],(18,3))[al_map2smpl]
                        apose[i][al_map2smpl==-1,:] = 0
                    except:
                        pass
                
                # reshape
                opose = np.reshape(opose,[-1,3])
                apose = np.reshape(apose,[-1,3])

                # make confidence 0 if apose and opose detections are far
                opose[np.sqrt((opose[:,0]-apose[:,0])**2 + (opose[:,1]-apose[:,1])**2) > hparams["data_kp_thres"],2] = 0
                apose[np.sqrt((opose[:,0]-apose[:,0])**2 + (opose[:,1]-apose[:,1])**2) > hparams["data_kp_thres"],2] = 0
                self.opose_res[cam].append(np.reshape(opose,[-1,24,3]))
                self.apose_res[cam].append(np.reshape(apose,[-1,24,3]))

        # image normalization
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) 
        
    def __len__(self):
        return len(self.opose_res["c0"])
    
    def __getitem__(self, idx):

        ########## overfit mode #########
        idx= 0 
        #################################

        # get full sequence length
        full_seq_len = self.data_lengths[idx]

        # get sequence start
        seq_start = np.random.randint(1,full_seq_len-self.seq_len-1)

        ############### overfit mode ###################
        seq_start = 1
        ###############################################

        # get full images paths
        full_img_pth = []
        for cam in ["c0","c1","c2","c3"]:
            full_img_pth_cam = [ospj(self.seq_dirs[cam][idx],"{:06d}.jpg".format(seq_start+x)) for x in range(self.seq_len)]
            full_img_pth.append(full_img_pth_cam)
        full_img_pth = np.array(full_img_pth)

        # get subject
        sub = full_img_pth[0,0].split("/")[-3]

        # get intrinsics
        cam_intr = np.stack([self.cam_intr["c0"][sub],
                        self.cam_intr["c1"][sub],
                        self.cam_intr["c2"][sub],
                        self.cam_intr["c3"][sub]]) 

        # get 2d keypoints
        j2d = torch.stack([torch.from_numpy(self.opose_res[cam][idx][seq_start:(seq_start + self.seq_len)]).float() for cam in ["c0","c1","c2","c3"]])

        ############### half the frames to get 25 FPS ########################
        full_img_pth = full_img_pth[:,::2]
        j2d = j2d[:,::2]

        # load, crop and scale
        images = torch.zeros(full_img_pth.shape[0],full_img_pth.shape[1],3,224,224).float()
        bbs = torch.zeros(full_img_pth.shape[0],full_img_pth.shape[1],3).float()
        for cam in range(full_img_pth.shape[0]):
            for t in range(full_img_pth.shape[1]):
                # load
                try:
                    img = cv2.imread(full_img_pth[cam,t])[:,:,::-1]/255.
                except:
                    import ipdb;ipdb.set_trace()

                # calculate cropping area
                if j2d[cam,t,:,2].all() == 0:
                    crp_scl_image = np.zeros([224,224,3])
                    scl = 0.
                    bb = torch.tensor([0.,0.]).float()
                else:
                    bb_min = j2d[cam,t,j2d[cam,t,:,2]!=0,:2].min(dim=0).values.int() - torch.randint(50,200,(1,2))
                    bb_min[bb_min<0] = 0
                    bb_min = bb_min.squeeze(0)
                    bb_max = j2d[cam,t,j2d[cam,t,:,2]!=0,:2].max(dim=0).values.int() + torch.randint(50,200,(1,2))
                    bb_max[bb_max>1000] = 1000
                    bb_max = bb_max.squeeze(0)

                    # crop and scale
                    crp_scl_image,scl,_ = resize_with_pad(img[bb_min[1]:bb_max[1],bb_min[0]:bb_max[0],:])

                    # calcuate crop,scale parameters (P in Airpose) 
                    bb = 1 - ((bb_min+bb_max)/2)/torch.from_numpy((self.cam_intr["c"+str(cam)][full_img_pth[cam,t].split("/")[-3]][:2,2])).float()
                
                bbs[cam,t] = torch.cat([bb,torch.tensor([scl]).float()])

                # normalize the image (for resnet)
                images[cam,t] = self.normalize(torch.from_numpy(crp_scl_image.transpose(2,0,1)).float())
                

        full_img_pth_list = [list(x) for x in list(full_img_pth)]
        

        return {"full_im_paths":full_img_pth_list, "images":images, "bbs":bbs, "j2d":j2d, "cam_intr":cam_intr}




        



