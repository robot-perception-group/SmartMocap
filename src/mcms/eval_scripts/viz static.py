import bpy
from mathutils import *
from human_body_prior.body_model.body_model import BodyModel
import numpy as np
import torch
import math
from matplotlib import cm

D = bpy.data
C = bpy.context

empty = D.objects.new("empty",None)
C.scene.collection.objects.link(empty)
cmap = cm.get_cmap('viridis')
# empty.rotation_euler[0] = math.radians(90)
# empty.location[2] = 1.16

data = np.load("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/savitr_test/0029/test_00000.npz")

bm = BodyModel("/home/nsaini/Datasets/smpl_models/smplh/neutral/model.npz")

smpl_out = {"0":data["verts"]}
try:
    cam_trans = data["cam_trans"][0]
    cam_rots = data["cam_rots"][0]
    num_cams = cam_trans.shape[0]
    cams_available = True
except:
    cams_available = False



motion_range = range(0,cam_trans.shape[1],4)
# motion_range = range(8,17)

for idx in motion_range:

    mat = bpy.data.materials.new("mat_"+str(idx))
    mat.diffuse_color = cmap(idx/cam_trans.shape[1])

    smpl_mesh = D.meshes.new("smpl_mesh"+str(idx))
    smpl_obj = D.objects.new(smpl_mesh.name,smpl_mesh)
    smpl_mesh.from_pydata(smpl_out[str(0)][idx],[],list(bm.f.detach().numpy()))
    smpl_mesh.materials.append(mat)
    smpl_obj.show_transparent = True
    C.scene.collection.objects.link(smpl_obj)
    smpl_obj.parent = empty

    if cams_available:
        bpy.ops.import_mesh.stl(filepath="/is/ps3/nsaini/projects/mcms/cam.stl")
        cam_stl_obj = [obj for obj in bpy.context.scene.objects if obj.name=="cam"]
        cam_stl_obj[0].name = "cam_0_"+str(idx)
        for n in range(1,num_cams):
            cam_stl_obj.append(cam_stl_obj[0].copy())
            cam_stl_obj[-1].name = "cam_"+str(n) + "_"+str(idx)
            bpy.context.collection.objects.link(cam_stl_obj[-1])
        for cam, cam_obj in enumerate(cam_stl_obj):
            cam_obj.parent = empty
            cam_obj.location[0] = cam_trans[cam,idx,0]
            cam_obj.location[1] = cam_trans[cam,idx,1]
            cam_obj.location[2] = cam_trans[cam,idx,2]
            cam_obj.scale[0] = 10
            cam_obj.scale[1] = 10
            cam_obj.scale[2] = 10
            cam_obj.rotation_mode = 'QUATERNION'
            cam_obj.rotation_quaternion = cam_rots[cam,0]
            cam_mesh = cam_obj.data
            cam_mesh.materials.append(mat)
            cam_obj.show_transparent = True