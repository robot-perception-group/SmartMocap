import torch
import logging
import os
import os.path as osp
import pickle as pkl
import json
from torch.utils.data import Dataset
import sys
import cv2
import numpy as np
from torchvision import transforms
from ..utils.utils import resize_with_pad
import copy
import torchgeometry as tgm

# remove nose as head
op_map2smpl = np.array([8,12,9,-1,13,10,-1,14,11,-1,19,22,1,-1,-1,-1,5,2,6,3,7,4,-1,-1])
al_map2smpl = np.array([-1,11,8,-1,12,9,-1,13,10,-1,-1,-1,1,-1,-1,-1,5,2,6,3,7,4,-1,-1])
dlc_map2smpl = np.array([-1,3,2,-1,4,1,-1,5,0,-1,-1,-1,-1,-1,-1,-1,9,8,10,7,11,6,-1,-1])

def get_copenet_real_traintest(datapath="/ps/project/datasets/AirCap_ICCV19/copenet_data",train_range=range(0,7000),test_range=range(8000,15000),shuffle_cams=False,first_cam=0,kp_agrmnt_threshold=100):
    train_dset = copenet_real(datapath,train_range,shuffle_cams,first_cam,kp_agrmnt_threshold)
    test_dset = copenet_real(datapath,test_range,shuffle_cams,first_cam,kp_agrmnt_threshold)
    return train_dset, test_dset

class copenet_real(Dataset):
    def __init__(self,hparams,drange:range,first_cam=0,kp_agrmnt_threshold=100):
        super().__init__()

        if osp.exists(hparams["data_datapath"]):
            print("loading copenet real data...")
            
            db_im1 = [osp.join(hparams["data_datapath"],"machine_1","images") + "/" + "{:06d}.jpg".format(i) for i in drange]
            db_im2 = [osp.join(hparams["data_datapath"],"machine_2","images") + "/" + "{:06d}.jpg".format(i) for i in drange]

            opose_m1 = pkl.load(open(osp.join(hparams["data_datapath"],"machine_1","openpose_res.pkl"),"rb"))
            opose_m2 = pkl.load(open(osp.join(hparams["data_datapath"],"machine_2","openpose_res.pkl"),"rb"))
            apose_m1 = json.load(open(osp.join(hparams["data_datapath"],"machine_1","alphapose_res.json"),"r"))
            apose_m2 = json.load(open(osp.join(hparams["data_datapath"],"machine_2","alphapose_res.json"),"r"))
            
            if drange[0] == 0:
                pass
            elif drange[0] == 8000:
                # dlc_file_m1 = osp.join(hparams["datapath"],"machine_1","yt-1-1DLC_resnet_101_fineMar5shuffle1_100000_filtered.h5")
                # dlc_file_m2 = osp.join(hparams["datapath"],"machine_2","yt-2-1DLC_resnet_101_fineMar5shuffle1_100000_filtered.h5")

                # with h5py.File(dlc_file_m1,"r") as fl:
                #     dlc_m1 = np.array([fl["df_with_missing/table"][i][1].reshape(-1,3) for i in range(7000)])
                # with h5py.File(dlc_file_m2,"r") as fl:
                #     dlc_m2 = np.array([fl["df_with_missing/table"][i][1].reshape(-1,3) for i in range(7000)])
                # self.raw_dlc0 = dlc_m1
                # self.raw_dlc1 = dlc_m2
                # self.mapped_dlc0 = self.raw_dlc0[:,dlc_map2smpl]
                # self.mapped_dlc1 = self.raw_dlc1[:,dlc_map2smpl]
                # self.mapped_dlc0[:,dlc_map2smpl==-1,:] = 0
                # self.mapped_dlc1[:,dlc_map2smpl==-1,:] = 0
                pass
            
            self.raw_apose0 = apose_m1
            self.raw_apose1 = apose_m2
            

            opose = np.zeros([2,len(drange),24,3])
            apose = np.zeros([2,len(drange),24,3])
            
            count = 0
            for i in drange:
                try:
                    opose[0,count] = opose_m1["{:06d}".format(i)]["pose"][0,op_map2smpl]
                    opose[0,count][op_map2smpl==-1,:] = 0
                except:
                    pass
                try:
                    apose[0,count] = np.reshape(apose_m1["{:06d}".format(i)]["people"][0]["pose_keypoints_2d"],(18,3))[al_map2smpl]
                    apose[0,count][al_map2smpl==-1,:] = 0
                except:
                    pass
                count += 1

            count = 0
            for i in drange:
                try:
                    opose[1,count] = opose_m2["{:06d}".format(i)]["pose"][0,op_map2smpl]
                    opose[1,count][op_map2smpl==-1,:] = 0
                except:
                    pass
                try:
                    apose[1,count] = np.reshape(apose_m2["{:06d}".format(i)]["people"][0]["pose_keypoints_2d"],(18,3))[al_map2smpl]
                    apose[1,count][al_map2smpl==-1,:] = 0
                except:
                    pass
                count += 1

            self.db = {}
            self.db["im0"] = db_im1
            self.db["im1"] = db_im2

            opose = np.reshape(opose,[-1,3])
            apose = np.reshape(apose,[-1,3])

            self.opose_smpl_fmt = np.reshape(opose,[2,-1,24,3])
            self.apose_smpl_fmt = np.reshape(apose,[2,-1,24,3])

            opose[np.sqrt((opose[:,0]-apose[:,0])**2 + (opose[:,1]-apose[:,1])**2) > kp_agrmnt_threshold,2] = 0
            apose[np.sqrt((opose[:,0]-apose[:,0])**2 + (opose[:,1]-apose[:,1])**2) > kp_agrmnt_threshold,2] = 0

            self.opose = np.reshape(opose,[2,-1,24,3])
            self.apose = np.reshape(apose,[2,-1,24,3])

            cv_file = cv2.FileStorage(osp.join(hparams["data_datapath"],"machine_1","camera_calib.yml"), cv2.FILE_STORAGE_READ)
            self.intr0 = cv_file.getNode("K").mat()
            cv_file.release()
            cv_file = cv2.FileStorage(osp.join(hparams["data_datapath"],"machine_2","camera_calib.yml"), cv2.FILE_STORAGE_READ)
            self.intr1 = cv_file.getNode("K").mat()
            cv_file.release()

            
            pose0 = pkl.load(open(osp.join(hparams["data_datapath"],"machine_1","markerposes_corrected_all.pkl"),"rb"))
            pose1 = pkl.load(open(osp.join(hparams["data_datapath"],"machine_2","markerposes_corrected_all.pkl"),"rb"))
            rvecs = []
            k0 = sorted(pose0.keys())
            for i in range(len(pose1)):
                try:
                    rvecs.append(pose0[k0[i]]["0"]["rvec"])
                except:
                    rvecs.append(np.zeros(3))
            rvecs = np.array(rvecs)
            self.extr0 = tgm.angle_axis_to_rotation_matrix(torch.from_numpy(rvecs).float())

            rvecs = []
            k1 = sorted(pose1.keys())
            for i in range(len(pose1)):
                try:
                    rvecs.append(pose1[k0[i]]["0"]["rvec"])
                except:
                    rvecs.append(np.zeros(3))
            rvecs = np.array(rvecs)
            self.extr1 = tgm.angle_axis_to_rotation_matrix(torch.from_numpy(rvecs).float())
            for i in range(len(pose1)):
                self.extr0[i,:3,3] = torch.from_numpy(pose0["{:06d}".format(i)]["0"]["tvec"]).float()  
                self.extr1[i,:3,3] = torch.from_numpy(pose1["{:06d}".format(i)]["0"]["tvec"]).float()
                

            self.num_cams = 2
            # self.shuffle_cams = shuffle_cams
            # if shuffle_cams:
            #     self.first_cam = -1
            # else:
            #     self.first_cam = first_cam
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

            # sequence length
            self.seq_len = hparams["data_seq_len"]

        else:
            sys.exit("invalid datapath !!")


    def __len__(self):
        return len(self.db["im0"])-self.seq_len


    def __getitem__(self, idx):


        # get sequence start
        seq_start = idx

        # get intrinsics
        cam_intr = np.stack([self.intr0,self.intr1],axis=0)

        # get 2d keypoints
        j2d = torch.from_numpy(self.opose[:,seq_start:seq_start+self.seq_len,:,:]).float()

        # load, crop and scale
        images = torch.zeros(2,self.seq_len,3,224,224).float()
        bbs = torch.zeros(2,self.seq_len,3).float()
        for cam in range(2):
            for t in range(self.seq_len):
                # load
                try:
                    img = cv2.imread(self.db["im"+str(cam)][idx+t])[:,:,::-1]/255.
                except:
                    import ipdb;ipdb.set_trace()

                # calculate cropping area
                if (j2d[cam,t,:,2]==0).all():
                    crp_scl_image = np.zeros([224,224,3])
                    scl = 0.
                    bb = torch.tensor([0.,0.]).float()
                    print("No keypoints found in image")
                else:
                    bb_min = j2d[cam,t,j2d[cam,t,:,2]!=0,:2].min(dim=0).values.int() - torch.randint(50,200,(1,2))
                    bb_min[bb_min<0] = 0
                    bb_min = bb_min.squeeze(0)
                    bb_max = j2d[cam,t,j2d[cam,t,:,2]!=0,:2].max(dim=0).values.int() + torch.randint(50,200,(1,2))
                    bb_max = bb_max.squeeze(0)
                    bb_max[0] = min(bb_max[0],img.shape[1])
                    bb_max[1] = min(bb_max[1],img.shape[0])

                    # crop and scale
                    crp_scl_image,scl,_ = resize_with_pad(img[bb_min[1]:bb_max[1],bb_min[0]:bb_max[0],:])

                    # calcuate crop,scale parameters (P in Airpose) 
                    bb = 1 - ((bb_min+bb_max)/2)/torch.from_numpy((cam_intr[cam,:2,2])).float()
                
                bbs[cam,t] = torch.cat([bb,torch.tensor([scl]).float()])

                # normalize the image (for resnet)
                images[cam,t] = self.normalize(torch.from_numpy(crp_scl_image.transpose(2,0,1)).float())

        full_img_pth_list = [self.db["im0"][idx:idx+self.seq_len],self.db["im1"][idx:idx+self.seq_len]]

        return {"full_im_paths":full_img_pth_list, "images":images, "bbs":bbs, "j2d":j2d, "cam_intr":cam_intr,
                "moshbetas":None, "moshorient":None, "moshtrans":None, "moshpose":None, "mosh_available":False}

