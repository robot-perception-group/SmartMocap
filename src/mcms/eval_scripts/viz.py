import enum
import bpy
from mathutils import *
from human_body_prior.body_model.body_model import BodyModel
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import copy

D = bpy.data
C = bpy.context

empty = D.objects.new("empty",None)
C.scene.collection.objects.link(empty)
# empty.rotation_euler[0] = math.radians(90)
# empty.location[2] = 1.16

# 114
# data = np.load("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/volley_data5/0000/stage_05/stitched_seq_start_00000.npz")
# 414
# data = np.load("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/copenet_7/0000/stage_02/_seq_start_03000.npz")
# 599
data = np.load("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/rich_first_and_moving_cam_zLatent_val_0228/0000/gt_and_res_strt_off_10.npz")


bm = BodyModel("/home/nsaini/Datasets/smpl_models/smplh/neutral/model.npz")

if len(data["verts"].shape) == 3:
    smpl_out = data["verts"][np.newaxis]
    try:
        cam_trans = data["cam_trans"][np.newaxis]
        cam_rots = data["cam_rots"][np.newaxis]
        num_cams = cam_trans.shape[1]
        cams_available = True
    except:
        cams_available = False
elif len(data["verts"].shape) == 4:
    smpl_out = data["verts"]
    try:
        cam_trans = data["cam_trans"]
        cam_rots = data["cam_rots"]
        num_cams = cam_trans.shape[1]
        cams_available = True
    except:
        cams_available = False



cam_stl_obj = []
if cams_available:
    for idx in range(smpl_out.shape[0]):
        for n in range(0,num_cams):
            bpy.ops.import_mesh.stl(filepath="/is/ps3/nsaini/projects/mcms/cam.stl")
            cam_stl_obj.append([obj for obj in bpy.context.scene.objects if obj.name=="cam"][0])
            cam_stl_obj[-1].name = "cam_{}_{}".format(idx,n)
            cam_stl_obj[-1].parent = empty

cmap = plt.get_cmap("Set1",10)


for idx in range(smpl_out.shape[0]):

    smpl_mesh = D.meshes.new("smpl_mesh"+str(idx))
    smpl_obj = D.objects.new(smpl_mesh.name,smpl_mesh)
    smpl_mesh.from_pydata(smpl_out[idx,0],[],list(bm.f.detach().numpy()))
    C.scene.collection.objects.link(smpl_obj)
    smpl_obj.parent = empty
    mat = bpy.data.materials.new("mat_"+str(idx))
    mat.diffuse_color = cmap(0)
    smpl_mesh.materials.append(mat)
    smpl_obj.show_transparent = True

    cam_mat = []
    if cams_available:
        for cam in range(num_cams):
            cam_obj = D.objects.get("cam_{}_{}".format(idx,cam))
            cam_obj.location[0] = cam_trans[idx,cam,0,0]
            cam_obj.location[1] = cam_trans[idx,cam,0,1]
            cam_obj.location[2] = cam_trans[idx,cam,0,2]
            cam_obj.scale[0] = 10
            cam_obj.scale[1] = 10
            cam_obj.scale[2] = 10
            cam_obj.rotation_mode = 'QUATERNION'
            cam_obj.rotation_quaternion = cam_rots[idx,cam,0]
            cam_mat.append(bpy.data.materials.new("cam_mat_{}_{}".format(idx,cam)))
            cam_mat[-1].diffuse_color = cmap(cam+1)
            cam_obj.data.materials.append(cam_mat[-1])
            cam_obj.show_transparent = True

def anim_handler(scene):
    frame=scene.frame_current
    
    for idx in range(smpl_out.shape[0]):
        ob = D.objects.get("smpl_mesh"+str(idx))
        ob.data.clear_geometry()
        ob.data.from_pydata(smpl_out[idx,frame],[],list(bm.f.detach().numpy()))

        if cams_available:
            for cam in range(num_cams):
                cam_obj = D.objects.get("cam_{}_{}".format(idx,cam))
                cam_obj.location[0] = cam_trans[idx,cam,frame,0]
                cam_obj.location[1] = cam_trans[idx,cam,frame,1]
                cam_obj.location[2] = cam_trans[idx,cam,frame,2]
                # cam_obj.rotation_mode = 'QUATERNION'
                cam_obj.rotation_quaternion = cam_rots[idx,cam,frame]

bpy.app.handlers.frame_change_pre.append(anim_handler)


def new_plane(mylocation, mysize, myname):
    bpy.ops.mesh.primitive_plane_add(
        size=mysize,
        calc_uvs=True,
        enter_editmode=False,
        align='WORLD',
        location=mylocation,
        rotation=(0, 0, 0),
        scale=(0, 0, 0))
    current_name = bpy.context.selected_objects[0].name
    plane = bpy.data.objects[current_name]
    plane.name = myname
    plane.data.name = myname + "_mesh"
    return


# for idi,i in enumerate(np.arange(-10.5,11.5)):
#     for idj,j in enumerate(np.arange(-10.5,11.5)):
#         new_plane((i,j,0),0.95,"plane_{}_{}".format(idi,idj))