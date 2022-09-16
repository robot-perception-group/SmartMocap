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

data = np.load("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/volley_data_teaser4/0000/stage_00/_seq_start_01910.npz")
# data = np.load("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/copenet_zLatent/0000/stage_01/_seq_start_00005.npz")

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



motion_range = range(0,cam_trans.shape[2],3)
# motion_range = range(660,760,10)
# motion_range = range(270,380,10)
# motion_range = [4,23]


for dat_idx in range(smpl_out.shape[0]):

    for idx in motion_range:

        offsets = np.array([0.05*idx,0,0])

        mat = bpy.data.materials.new("mat_"+str(idx))
        mat.diffuse_color = cmap(1 - (idx-motion_range[0])/(motion_range[-1]-motion_range[0]))
        # mat.diffuse_color = cmap(0.2)

        smpl_mesh = D.meshes.new("smpl_mesh"+str(idx))
        smpl_obj = D.objects.new(smpl_mesh.name,smpl_mesh)
        # smpl_mesh.from_pydata(smpl_out[dat_idx,idx] + offsets,[],list(bm.f.detach().numpy()))
        smpl_mesh.from_pydata(smpl_out[dat_idx,idx],[],list(bm.f.detach().numpy()))
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
                cam_obj.location[0] = cam_trans[dat_idx,cam,idx,0] #+ offsets[0]
                cam_obj.location[1] = cam_trans[dat_idx,cam,idx,1]
                cam_obj.location[2] = cam_trans[dat_idx,cam,idx,2]
                cam_obj.scale[0] = 10
                cam_obj.scale[1] = 10
                cam_obj.scale[2] = 10
                cam_obj.rotation_mode = 'QUATERNION'
                cam_obj.rotation_quaternion = cam_rots[dat_idx,cam,idx]
                cam_mesh = cam_obj.data
                cam_mesh.materials.append(mat)
                cam_obj.show_transparent = True



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
#     for idj,j in enumerate(np.arange(-15.5,21.5)):
#         new_plane((i,j,0),0.95,"plane_{}_{}".format(idi,idj))



# set camera pose
render_cam = [obj for obj in bpy.context.scene.objects if obj.name=="Camera"][0]
render_cam.rotation_mode = 'XYZ'
render_cam.location.x = -0.23
render_cam.location.y = 6.79
render_cam.location.z = 2.55
render_cam.rotation_euler[0] = 75.1 * (math.pi/180.0)
render_cam.rotation_euler[1] = 0.816 * (math.pi/180.0)
render_cam.rotation_euler[2] = -178 * (math.pi/180.0)

# delete point light
point_light_obj = [obj for obj in bpy.context.scene.objects if obj.name=="Light"][0]
# point_light_obj.data.energy = 20
# point_light_obj.location.x = 0.09
# point_light_obj.location.y = 0.865
# point_light_obj.location.z = 0.885
bpy.data.objects.remove(point_light_obj,do_unlink=True)

# add area light
light_data = bpy.data.lights.new(name="area_light", type='AREA')
light_data.energy = 2000
light_data.shape = 'RECTANGLE'
light_data.size = 20
light_data.size_y = 20

light_obj = bpy.data.objects.new(name="area_light", object_data=light_data)

bpy.context.collection.objects.link(light_obj)
light_obj.location.x = 0
light_obj.location.y = 0
light_obj.location.z = 10

bpy.ops.import_scene.fbx(filepath="/is/ps3/nsaini/projects/mcms/axis.fbx")