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

data = np.load("/is/ps3/nsaini/projects/mcms/temp.npy")

bm = BodyModel("/home/nsaini/Datasets/smpl_models/smplh/neutral/model.npz")

smpl_out = {"0":data}

motion_range = [0]
# motion_range = range(8,17)

for idx in motion_range:

    smpl_mesh = D.meshes.new("smpl_mesh"+str(idx))
    smpl_obj = D.objects.new(smpl_mesh.name,smpl_mesh)
    smpl_mesh.from_pydata(smpl_out[str(idx)][0],[],list(bm.f.detach().numpy()))
    C.scene.collection.objects.link(smpl_obj)
    smpl_obj.parent = empty
    smpl_obj.location[0] = idx

def anim_handler(scene):
    frame=scene.frame_current
    
    for idx in motion_range:
        ob = D.objects.get("smpl_mesh"+str(idx))
        ob.data.clear_geometry()
        ob.data.from_pydata(smpl_out[str(idx)][frame],[],list(bm.f.detach().numpy()))

bpy.app.handlers.frame_change_pre.append(anim_handler)