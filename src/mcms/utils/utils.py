import cv2
import torch
from pytorch3d.transforms import rotation_conversions as p3dt

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