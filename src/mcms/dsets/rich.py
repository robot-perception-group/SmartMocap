from wsgiref.handlers import read_environ
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
import copy
from torchvision import transforms
from scipy.spatial.transform import Rotation

from ..utils.utils import resize_with_pad

# remove nose as head
op_map2smpl = np.array([8,-1,-1,-1,13,10,-1,14,11,-1,19,22,-1,-1,-1,-1,5,2,6,3,7,4,-1,-1])
al_map2smpl = np.array([-1,11,8,-1,12,9,-1,13,10,-1,-1,-1,1,-1,-1,-1,5,2,6,3,7,4,-1,-1])
smpl_map2op = np.array([-1,12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            dtype=np.int32) # Nose replaced with head, but weight should be 0
# mapping from Vassilis's code
[24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

class rich(Dataset):
    def __init__(self,datapath, used_cams=None):
        super().__init__()
        
        # camera calibration files
        cams_calibs = sorted(glob.glob(ospj(datapath,"calibration/*")))
        if used_cams is not None:
            self.used_cams = sorted(used_cams)
        else:
            self.used_cams = used_cams

        self.num_cams = len(cams_calibs) + 1

        cam_intrinsics = np.stack([extract_cam_param_xml(cams_calibs[i])[0] for i in range(len(cams_calibs))])
        self.cam_extrinsics = np.stack([extract_cam_param_xml(cams_calibs[i])[1] for i in range(len(cams_calibs))])
        # copy intrinsics from first cam to last cam (freecam)
        self.cam_intrinsics = np.concatenate([cam_intrinsics, np.expand_dims(cam_intrinsics[0],axis=0)],axis=0)

        self.frames_dirs = sorted(glob.glob(ospj(datapath,"data/*")))
        self.n_frames = len(self.frames_dirs)

        # imageresolution
        self.im_res = []
        for c in range(self.num_cams-1):
            self.im_res.append(cv2.imread(ospj(self.frames_dirs[0],"00","images_orig",self.frames_dirs[0].split("/")[-1]+"_{:02d}.png".format(c))).shape[:2])
        self.im_res.append(cv2.imread(ospj(self.frames_dirs[0],"00","freecam",self.frames_dirs[0].split("/")[-1]+"_{:02d}_images_orig.png".format(10))).shape[:2])
        self.im_res = np.array(self.im_res)
        
    def __len__(self):
        return self.n_frames
    
    def __getitem__(self, idx, seq_len=25):


        # load keypoints for static cams
        j2d = []
        full_im_path_list = []
        pare_res = []
        pare_cams = []
        for c in range(self.num_cams-1):
            kps = []
            im_paths = []
            pare_results_cam = []
            pare_cams_res = []
            for d in self.frames_dirs[idx:idx+seq_len]:
                try:
                    bbox = json.load(open(ospj(d,"00","bbox_refine",d.split("/")[-1]+"_{:02d}.json".format(c)),"r"))
                    bbox_top_left = np.array([bbox["x1"],bbox["y1"]])

                    keypoints = json.load(open(ospj(d,"00","keypoints_refine",d.split("/")[-1]+"_{:02d}_keypoints.json".format(c)),"r"))
                    keypoints = np.array(keypoints['people'][0]['pose_keypoints_2d']).reshape([-1, 3])
                    keypoints[:,:2] = keypoints[:,:2] + bbox_top_left
                    keypoints = keypoints[op_map2smpl]
                    keypoints[op_map2smpl==-1,:] = 0
                    
                    kps.append(keypoints)
                except:
                    print("No keypoints for "+ospj(d,"00","keypoints_refine",d.split("/")[-1]+"_{:02d}_keypoints.json".format(c)))
                    kps.append(np.zeros([24,3]))

                try:
                    # load pare results
                    cams_present = sorted([int(x.split("/")[-1].split(".")[0].split("_")[-1]) for x in glob.glob(ospj(d,"00","pare_results_refine","*.pkl"))])
                    pare_pkl = pkl.load(open(ospj(d,"00","pare_results_refine",d.split("/")[-1]+"_{:02d}.pkl".format(c)),"rb"))
                    
                    pare_cam = convert_pare_to_full_img_cam(pare_pkl["pred_cam"][cams_present.index(c)].unsqueeze(0),
                                    bbox["y2"]-bbox["y1"],
                                    torch.tensor([(bbox["x1"]+bbox["x2"])/2,(bbox["y1"]+bbox["y2"])/2]).type_as(pare_pkl["pred_cam"]).unsqueeze(0),
                                    self.im_res[cams_present.index(c),1],
                                    self.im_res[cams_present.index(c),0],
                                    self.cam_intrinsics[cams_present.index(c),0,0]+self.cam_intrinsics[cams_present.index(c),1,1]/2)
                    pare_cams_res.append(pare_cam.detach().cpu())
                    pare_pose = pare_pkl["pred_pose"][:,:24]
                    pare_results_cam.append(pare_pose[cams_present.index(c)].detach().cpu())
                    # pare_res.append(np.stack([(Rotation.from_matrix(pare_resuls[:,i].detach().cpu().numpy())).mean().as_rotvec() for i in range(pare_resuls.shape[1])]))

                except:
                    print("No PARE results for "+ospj(d,"00","pare_results_refine",d.split("/")[-1]+"_{:02d}.pkl".format(c)))
                    if len(pare_results_cam) == 0:
                        pare_results_cam.append(torch.eye(3).repeat(24,1,1).float())
                        pare_cams_res.append(torch.zeros(1,3).float())
                    else:
                        pare_results_cam.append(copy.deepcopy(pare_results_cam[-1]))
                        pare_cams_res.append(copy.deepcopy(pare_cams_res[-1]))
                
                im_paths.append(ospj(d,"00","images_orig",d.split("/")[-1]+"_{:02d}.png".format(c)))

            j2d.append(np.stack(kps))
            full_im_path_list.append(im_paths)
            pare_res.append(torch.stack(pare_results_cam))
            pare_cams.append(torch.cat(pare_cams_res))
        
        # add keypoints for freecam
        kps = []
        im_paths = []
        pare_results_cam = []
        pare_cams_res = []
        for d in self.frames_dirs[idx:idx+seq_len]:
            try:
                bbox = json.load(open(ospj(d,"00","freecam",d.split("/")[-1]+"_{:02d}_bbox.json".format(10)),"r"))
                bbox_top_left = np.array([bbox["x1"],bbox["y1"]])

                keypoints = json.load(open(ospj(d,"00","freecam_keypoints",d.split("/")[-1]+"_{:02d}_images_keypoints.json".format(10)),"r"))
                keypoints = np.array(keypoints['people'][0]['pose_keypoints_2d']).reshape([-1, 3])
                keypoints[:,:2] = keypoints[:,:2] + bbox_top_left
                keypoints = keypoints[op_map2smpl]
                keypoints[op_map2smpl==-1,:] = 0

                kps.append(keypoints)

                # load pare results
                pare_pkl = pkl.load(open(ospj(d,"00","freecam",d.split("/")[-1]+"_{:02d}_pare.pkl".format(10)),"rb"))
                pare_pose = pare_pkl["pred_pose"][:,:24]
                pare_results_cam.append(pare_pose[0].detach().cpu())

                pare_cam = convert_pare_to_full_img_cam(pare_pkl["pred_cam"][-1].unsqueeze(0),
                                    bbox["y2"]-bbox["y1"],
                                    torch.tensor([(bbox["x1"]+bbox["x2"])/2,(bbox["y1"]+bbox["y2"])/2]).type_as(pare_pkl["pred_cam"]).unsqueeze(0),
                                    self.im_res[-1,1],
                                    self.im_res[-1,0],
                                    self.cam_intrinsics[7,0,0]+self.cam_intrinsics[7,1,1]/2)
                pare_cams_res.append(pare_cam.detach().cpu())

            except:
                import ipdb;ipdb.set_trace()
                kps.append(np.zeros([24,3]))
                pare_results_cam.append(copy.deepcopy(pare_results_cam[-1]))
                pare_cams_res.append(copy.deepcopy(pare_cams_res[-1]))

            im_paths.append(ospj(d,"00","freecam",d.split("/")[-1]+"_{:02d}_images_orig.png".format(10)))


        
        j2d.append(np.stack(kps))
        full_im_path_list.append(im_paths)
        pare_res.append(torch.stack(pare_results_cam))
        pare_cams.append(torch.cat(pare_cams_res))
        pare_res = torch.stack(pare_res).detach().cpu().numpy()
        pare_cams = torch.stack(pare_cams).detach()


        pare_results = []
        for i in range(seq_len):
            pare_results.append(np.stack([Rotation.from_matrix(pare_res[:,i,j]).mean().as_rotvec() for j in range(1,pare_res.shape[2])]))

        pare_results = torch.from_numpy(np.stack(pare_results)).float()

        pare_res_orient = torch.from_numpy(np.stack([Rotation.from_matrix(pare_res[i,:,0]).as_rotvec() for i in range(pare_res.shape[0])])).float()
        
        j2d = torch.from_numpy(np.stack(j2d)).float()


        # load GT
        gt_trans = []
        gt_global_orient = []
        gt_body_pose = []
        gt_betas = []
        for n,d in enumerate(self.frames_dirs[idx:idx+seq_len]):
            gt_pth = ospj("/".join(d.split("/")[:-2]),"params","{:05d}".format(idx+n),"00","results_smpl","000.pkl")
            gt = pkl.load(open(gt_pth,"rb"))
            gt_betas.append(gt["betas"].detach().cpu())
            gt_trans.append(gt["transl"].detach().cpu())
            gt_global_orient.append(gt["global_orient"].detach().cpu())
            gt_body_pose.append(gt["body_pose"].detach().cpu())
        gt_trans = torch.cat(gt_trans)
        gt_global_orient = torch.cat(gt_global_orient)
        gt_body_pose = torch.cat(gt_body_pose)
        gt_betas = torch.cat(gt_betas)
        
        if self.used_cams is not None:
            return {"full_im_paths":np.array(full_im_path_list)[self.used_cams].tolist(), "j2d":j2d[self.used_cams], "cam_intr":self.cam_intrinsics[self.used_cams], 
                    "pare_poses":pare_results, "pare_orient":pare_res_orient[self.used_cams],
                    "gt_trans":gt_trans, "gt_global_orient":gt_global_orient,
                    "gt_body_pose":gt_body_pose,"gt_betas":gt_betas,
                    "j2d_op":j2d[self.used_cams][:,:,smpl_map2op],
                    "cam_extr": self.cam_extrinsics,
                    "pare_cams":pare_cams[self.used_cams]}
        else:    
            return {"full_im_paths":full_im_path_list, "j2d":j2d, "cam_intr":self.cam_intrinsics, 
                    "pare_poses":pare_results, "pare_orient":pare_res_orient,
                    "gt_trans":gt_trans, "gt_global_orient":gt_global_orient,
                    "gt_body_pose":gt_body_pose,"gt_betas":gt_betas,
                    "j2d_op":j2d[:,:,smpl_map2op],
                    "cam_extr": self.cam_extrinsics,
                    "pare_cams":pare_cams}






def extract_cam_param_xml(xml_path:str='', dtype=np.float):
    
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)

    extrinsics_mat = [float(s) for s in tree.find('./CameraMatrix/data').text.split()]
    intrinsics_mat = [float(s) for s in tree.find('./Intrinsics/data').text.split()]
    distortion_vec = [float(s) for s in tree.find('./Distortion/data').text.split()]

    focal_length_x = intrinsics_mat[0]
    focal_length_y = intrinsics_mat[4]
    center = np.array([[intrinsics_mat[2], intrinsics_mat[5]]], dtype=dtype)
    
    rotation = np.array([[extrinsics_mat[0], extrinsics_mat[1], extrinsics_mat[2]], 
                            [extrinsics_mat[4], extrinsics_mat[5], extrinsics_mat[6]], 
                            [extrinsics_mat[8], extrinsics_mat[9], extrinsics_mat[10]]], dtype=dtype)

    translation = np.array([[extrinsics_mat[3], extrinsics_mat[7], extrinsics_mat[11]]], dtype=dtype)

    extrinsics = np.array([[extrinsics_mat[0], extrinsics_mat[1], extrinsics_mat[2], extrinsics_mat[3]], 
                            [extrinsics_mat[4], extrinsics_mat[5], extrinsics_mat[6], extrinsics_mat[7]], 
                            [extrinsics_mat[8], extrinsics_mat[9], extrinsics_mat[10], extrinsics_mat[11]],
                            [0,0,0,1]], dtype=dtype)

    # t = -Rc --> c = -R^Tt
    cam_center = [  -extrinsics_mat[0]*extrinsics_mat[3] - extrinsics_mat[4]*extrinsics_mat[7] - extrinsics_mat[8]*extrinsics_mat[11],
                    -extrinsics_mat[1]*extrinsics_mat[3] - extrinsics_mat[5]*extrinsics_mat[7] - extrinsics_mat[9]*extrinsics_mat[11], 
                    -extrinsics_mat[2]*extrinsics_mat[3] - extrinsics_mat[6]*extrinsics_mat[7] - extrinsics_mat[10]*extrinsics_mat[11]]

    cam_center =np.array([cam_center], dtype=dtype)

    k1 = np.array([distortion_vec[0]], dtype=dtype)
    k2 = np.array([distortion_vec[1]], dtype=dtype)

    intrinsics = np.array([[focal_length_x, 0, center[0][0]], [0, focal_length_y, center[0][1]], [0, 0, 1]], dtype=dtype)

    return intrinsics, extrinsics, rotation, translation, cam_center, k1, k2


def convert_pare_to_full_img_cam(
        pare_cam, bbox_height, bbox_center,
        img_w, img_h, focal_length, crop_res=224):
    # Converts weak perspective camera estimated by PARE in
    # bbox coords to perspective camera in full image coordinates
    # from https://arxiv.org/pdf/2009.06549.pdf
    # taken from PARE code
    s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
    tz = 2 * focal_length / (bbox_height * s)

    cx = 2 * (bbox_center[:, 0] - (img_w / 2.)) / (s * bbox_height)
    cy = 2 * (bbox_center[:, 1] - (img_h / 2.)) / (s * bbox_height)

    cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)

    return cam_t

def convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height):
    '''
    Convert predicted camera from cropped image coordinates
    to original image coordinates
    :param cam (ndarray, shape=(3,)): weak perspective camera in cropped img coordinates
    :param bbox (ndarray, shape=(4,)): bbox coordinates (c_x, c_y, h)
    :param img_width (int): original image width
    :param img_height (int): original image height
    :return:
    '''
    cx, cy, h = bbox[:,0], bbox[:,1], bbox[:,2]
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[:,0] * (1. / (img_width / h))
    sy = cam[:,0] * (1. / (img_height / h))
    tx = ((cx - hw) / hw / sx) + cam[:,1]
    ty = ((cy - hh) / hh / sy) + cam[:,2]
    orig_cam = np.stack([sx, sy, tx, ty]).T
    return orig_cam

