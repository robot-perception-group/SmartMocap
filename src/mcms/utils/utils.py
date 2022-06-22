import cv2
import torch
from pytorch3d.transforms import rotation_conversions as p3dt
import numpy as np

def resize_with_pad(img,size=224):
    '''
    size: (Int) output would be size x size
    '''
    if img.shape[0] > img.shape[1]:
        biggr_dim = img.shape[0]
    else:
        biggr_dim = img.shape[1]
    scale = size/biggr_dim
    out_img = cv2.resize(img,(int(scale*img.shape[1]),int(scale*img.shape[0])))
    pad_top = (size - out_img.shape[0])//2
    pad_bottom = size - out_img.shape[0] - pad_top
    pad_left = (size - out_img.shape[1])//2
    pad_right = size - out_img.shape[1] - pad_left
    out_img = cv2.copyMakeBorder(out_img,
                                    pad_top,
                                    pad_bottom,
                                    pad_left,
                                    pad_right,
                                    cv2.BORDER_CONSTANT)

    return out_img, scale, [pad_left,pad_top]


def transform_smpl(trans_mat,smplvertices=None,smpljoints=None, orientation=None, smpltrans=None):
    if smplvertices is not None:
        verts =  torch.bmm(trans_mat[:,:3,:3],smplvertices.permute(0,2,1)).permute(0,2,1) +\
                    trans_mat[:,:3,3].unsqueeze(1)
    else:
        verts = None
    if smpljoints is not None:
        joints = torch.bmm(trans_mat[:,:3,:3],smpljoints.permute(0,2,1)).permute(0,2,1) +\
                         trans_mat[:,:3,3].unsqueeze(1)
    else:
        joints = None
    
    if smpltrans is not None:
        trans = torch.bmm(trans_mat[:,:3,:3],smpltrans.unsqueeze(2)).squeeze(2) +\
                         trans_mat[:,:3,3]
    else:
        trans = None

    if orientation is not None:
        orient = torch.bmm(trans_mat[:,:3,:3],orientation)
    else:
        orient = None    
    return verts, joints, orient, trans

def rottrans2transf(rotmat,trans):
    batch_size = rotmat.shape[0]
    assert rotmat.shape[0]==trans.shape[0], "rotmat and trans should have same batch size"
    return torch.cat([torch.cat([rotmat,trans.unsqueeze(2)],dim=2),torch.tensor([0,0,0,1]).unsqueeze(0).unsqueeze(0).float().repeat(batch_size,1,1).to(rotmat.device)],dim=1)

def smpl2nmg(poses,bm):
    joints = bm.forward(root_orient=poses[:,3:6],pose_body=poses[:,6:]).Jtr
    joint_pos_wrt_root = joints[:,1:] - joints[:,0:1]
    pose_angles = poses[:,3:].view(poses.shape[0],-1,3)

    temptrans = [rottrans2transf(p3dt.axis_angle_to_matrix(pose_angles[:,0]),joints[:,0])]
    for j in range(1,22):
        temptrans.append(rottrans2transf(torch.matmul(temptrans[bm.kintree_table[0,j]][:,:3,:3],p3dt.axis_angle_to_matrix(pose_angles[:,j])),joint_pos_wrt_root[:,j]))  
    transfs = torch.stack(temptrans,dim=1)
    transfs[:,:,:3,3] += poses[:,:3].unsqueeze(1)

    nmg_transfs = torch.zeros(poses.shape[0],22,9).float().to(poses.device)
    nmg_transfs[:,:,:6] = p3dt.matrix_to_rotation_6d(transfs[:,:,:3,:3])
    nmg_transfs[:,:,6:] = transfs[:,:,:3,3]

    return nmg_transfs

def nmg2smpl(nmg_transfs,bm):
    transfs = torch.zeros(nmg_transfs.shape[0],nmg_transfs.shape[1],4,4).float().to(nmg_transfs.device)
    transfs[:,:,:3,:3] = p3dt.rotation_6d_to_matrix(nmg_transfs[:,:,:6])
    transfs[:,:,:3,3] = nmg_transfs[:,:,6:]
    poses_angles = torch.zeros(transfs.shape[0],transfs.shape[1],3).float().to(nmg_transfs.device)
    
    for j in range(21,0,-1):
        poses_angles[:,j] = p3dt.matrix_to_axis_angle(torch.matmul(torch.inverse(transfs[:,bm.kintree_table[0,j],:3,:3]),transfs[:,j,:3,:3]))
    poses_angles[:,0] = p3dt.matrix_to_axis_angle(transfs[:,0,:3,:3])

    joints = bm.forward(root_orient=poses_angles[:,0],pose_body=poses_angles.view(poses_angles.shape[0],22*3)[:,3:]).Jtr

    trans = transfs[:,0,:3,3] - joints[:,0]

    return torch.cat([trans,poses_angles.reshape(trans.shape[0],22*3)],dim=1)


def to_homogeneous(rot,pos):

    tfm_rot = rot.reshape(-1,3,3)
    tfm_pos = pos.reshape(-1,3,1)
    tfm = torch.cat([tfm_rot,tfm_pos],dim=2)
    tfm = torch.cat([tfm,torch.tensor([0,0,0,1]).type_as(tfm).repeat(tfm.shape[0],1,1)],dim=1).reshape(*rot.shape[:-2],4,4)

    return tfm

def gmcclure(e,sigma):
    return e**2/(e**2+sigma**2)

# SMPL2openpose mapping from Vassilis's code
def smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
                     use_face_contour=False, openpose_format='coco25'):
    ''' Returns the indices of the permutation that maps OpenPose to SMPL
        Parameters
        ----------
        model_type: str, optional
            The type of SMPL-like model that is used. The default mapping
            returned is for the SMPLX model
        use_hands: bool, optional
            Flag for adding to the returned permutation the mapping for the
            hand keypoints. Defaults to True
        use_face: bool, optional
            Flag for adding to the returned permutation the mapping for the
            face keypoints. Defaults to True
        use_face_contour: bool, optional
            Flag for appending the facial contour keypoints. Defaults to False
        openpose_format: bool, optional
            The output format of OpenPose. For now only COCO-25 and COCO-19 is
            supported. Defaults to 'coco25'
    '''
    if openpose_format.lower() == 'coco25':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59,
                                     60, 61, 62], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 63, 22, 23, 24, 64,
                                          25, 26, 27, 65, 31, 32, 33, 66, 28,
                                          29, 30, 67], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 68, 37, 38, 39, 69,
                                          40, 41, 42, 70, 46, 47, 48, 71, 43,
                                          44, 45, 72], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                     63, 64, 65], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                          67, 28, 29, 30, 68, 34, 35, 36, 69,
                                          31, 32, 33, 70], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                          43, 44, 45, 73, 49, 50, 51, 74, 46,
                                          47, 48, 75], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    elif openpose_format == 'coco19':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8,
                             1, 4, 7, 25, 26, 27, 28],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 57, 22, 23, 24, 58,
                                          25, 26, 27, 59, 31, 32, 33, 60, 28,
                                          29, 30, 61], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 62, 37, 38, 39, 63,
                                          40, 41, 42, 64, 46, 47, 48, 65, 43,
                                          44, 45, 66], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 60, 25, 26, 27,
                                          61, 28, 29, 30, 62, 34, 35, 36, 63,
                                          31, 32, 33, 64], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 65, 40, 41, 42, 66,
                                          43, 44, 45, 67, 49, 50, 51, 68, 46,
                                          47, 48, 69], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(70, 70 + 51 +
                                         17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    else:
        raise ValueError('Unknown joint format: {}'.format(openpose_format))



def proper_smpl_fwd(smpl,root_orient,pose_body,trans,betas):
    smpl_out = smpl.forward(pose_body=pose_body,betas=betas)
    transf_mat = to_homogeneous(p3dt.axis_angle_to_matrix(root_orient),trans)
    v,j3d,_,_ = transform_smpl(transf_mat,smplvertices=smpl_out.v,smpljoints=smpl_out.Jtr)
    smpl_out.v = v
    smpl_out.Jtr = j3d
    return smpl_out


def get_norm_poses(poses, trans):

    # translation starting from zero (x and y)
    poses_matrix = p3dt.axis_angle_to_matrix(poses.view([poses.shape[0],-1,3]))
    # trans[:,[0,2]] -= trans[0,[0,2]]
    # import ipdb;ipdb.set_trace()
    fwd = poses_matrix[0,0,:3,2].clone()
    fwd[2] = 0
    fwd /= torch.linalg.norm(fwd)
    if fwd[0] > 0:
        tfm = p3dt.axis_angle_to_matrix(torch.tensor([0,0,torch.arccos(fwd[1])]).type_as(fwd).unsqueeze(0))
    else:
        tfm = p3dt.axis_angle_to_matrix(torch.tensor([0,0,-torch.arccos(fwd[1])]).type_as(fwd).unsqueeze(0))
    
    tfmd_orient = torch.matmul(tfm,poses_matrix[:,0])
    tfmd_trans = torch.matmul(tfm,trans.unsqueeze(2)).squeeze(2)
    
    poses_matrix[:,0] = tfmd_orient
    
    norm_poses = torch.cat([tfmd_trans,p3dt.matrix_to_axis_angle(poses_matrix).reshape(tfmd_trans.shape[0],-1)],dim=1)

    return norm_poses, tfm