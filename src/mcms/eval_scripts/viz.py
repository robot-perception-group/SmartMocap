import bpy
from mathutils import *
from human_body_prior.body_model.body_model import BodyModel
import numpy as np
import torch
import math

D = bpy.data
C = bpy.context


empty = D.objects.new("empty",None)
C.scene.collection.objects.link(empty)
# empty.rotation_euler[0] = math.radians(90)
# empty.location[2] = 1.16

data = np.load("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/test/0029/test_00451.npz")

bm = BodyModel("/home/nsaini/Datasets/smpl_models/smplh/neutral/model.npz")

smpl_out = {"0":data["verts"]}

try:
    cam_trans = data["cam_trans"][0]
    cam_rots = data["cam_rots"][0]
    num_cams = cam_trans.shape[0]
    cams_available = True
except:
    cams_available = False

motion_range = [0]
# motion_range = range(8,17)

for idx in motion_range:

    smpl_mesh = D.meshes.new("smpl_mesh"+str(idx))
    smpl_obj = D.objects.new(smpl_mesh.name,smpl_mesh)
    smpl_mesh.from_pydata(smpl_out[str(idx)][0],[],list(bm.f.detach().numpy()))
    C.scene.collection.objects.link(smpl_obj)
    smpl_obj.parent = empty
    smpl_obj.location[0] = idx

    if cams_available:
        for cam in range(num_cams):
            cam_mesh = D.cameras.new(name="cam_"+str(cam))
            cam_obj = D.objects.new(cam_mesh.name,cam_mesh)
            C.scene.collection.objects.link(cam_obj)
            cam_obj.parent = empty
            cam_obj.location[0] = cam_trans[cam,0,0]
            cam_obj.location[1] = cam_trans[cam,0,1]
            cam_obj.location[2] = cam_trans[cam,0,2]
            cam_obj.rotation_mode = 'QUATERNION'
            cam_obj.rotation_quaternion = cam_rots[cam,0]

def anim_handler(scene):
    frame=scene.frame_current
    
    for idx in motion_range:
        ob = D.objects.get("smpl_mesh"+str(idx))
        ob.data.clear_geometry()
        ob.data.from_pydata(smpl_out[str(idx)][frame],[],list(bm.f.detach().numpy()))

        if cams_available:
            for cam in range(num_cams):
                cam_obj = D.objects.get("cam_"+str(cam))
                cam_obj.location[0] = cam_trans[cam,frame,0]
                cam_obj.location[1] = cam_trans[cam,frame,1]
                cam_obj.location[2] = cam_trans[cam,frame,2]
                cam_obj.rotation_mode = 'QUATERNION'
                cam_obj.rotation_quaternion = cam_rots[cam,idx]

bpy.app.handlers.frame_change_pre.append(anim_handler)