import numpy as np
from human_body_prior.body_model.body_model import BodyModel
import torch
from pytorch3d import transforms
import pickle as pkl
from matplotlib import cm

device = "cuda"

dataset = "train"
if dataset == "train":
        viz_dir = "train_data"
        res_id = 1
else:
    viz_dir = "test_data"
    res_id = 0

################ Baseline ###################################
fname = "/is/ps3/nsaini/projects/copenet_real/copenet_logs/hmr/version_2_from_newlytrinedckpt/checkpoints/epoch=388.pkl"
fname0 = fname + "0"
fname1 = fname + "1"
res0 = pkl.load(open(fname0,"rb"))
res1 = pkl.load(open(fname1,"rb"))
bl_smpl_angles0 = torch.cat([i["output"]["pred_angles"].to("cuda") for i in res0[res_id]])
bl_smpl_rotmat0 = transforms.rotation_conversions.axis_angle_to_matrix(bl_smpl_angles0)
bl_smpl_wrt_cam0 = torch.eye(4,device=device).float().unsqueeze(0).expand([6990,-1,-1]).clone()
bl_smpl_wrt_cam0[:,:3,:3] = bl_smpl_rotmat0[:,0]
bl_smpl_wrt_cam0[:,:3,3] = torch.cat([i["output"]["pred_smpltrans"].to(device) for i in res0[res_id]])
bl_smpl_angles1 = torch.cat([i["output"]["pred_angles"].to("cuda") for i in res1[res_id]])
bl_smpl_rotmat1 = transforms.rotation_conversions.axis_angle_to_matrix(bl_smpl_angles1)
bl_smpl_wrt_cam1 = torch.eye(4,device=device).float().unsqueeze(0).expand([6990,-1,-1]).clone()
bl_smpl_wrt_cam1[:,:3,:3] = bl_smpl_rotmat1[:,0]
bl_smpl_wrt_cam1[:,:3,3] = torch.cat([i["output"]["pred_smpltrans"].to(device) for i in res1[res_id]])
bl_cam1_wrt_smpl = torch.inverse(bl_smpl_wrt_cam1)
bl_cam1_wrt_cam0 = torch.matmul(bl_smpl_wrt_cam0,bl_cam1_wrt_smpl)
bl_pred_vertices_cam0 = torch.cat([i["output"]["tr_pred_vertices_cam"].to("cuda") for i in res0[res_id]]).detach().cpu().numpy()
bl_pred_vertices_cam1 = torch.cat([i["output"]["tr_pred_vertices_cam"].to("cuda") for i in res1[res_id]]).detach().cpu().numpy()

################# Airpose ##################################

# fname = "/is/cluster/nsaini/copenet_logs/copenet_twoview_newcorrectedruns/copenet_twoview_newcorrectedruns/checkpoints/epoch-257.pkl"
# res = pkl.load(open(fname,"rb"))
# airpose_pred_vertices_cam0 = (torch.cat([i["output"]["pred_vertices_cam0"].to("cuda") for i in res[res_id]])).detach().cpu().numpy()
# airpose_pred_vertices_cam1 = (torch.cat([i["output"]["pred_vertices_cam1"].to("cuda") for i in res[res_id]])).detach().cpu().numpy()

################# AirPose+ ############################################

if dataset == "train":
    res = np.load("/is/ps3/nsaini/Thesis/AirPoseRAL_mat/train_pl_verts0_RAL.npz")
else:
    res = np.load("/is/ps3/nsaini/Thesis/AirPoseRAL_mat/test_pl_verts_RAL.npz")
smplx_model = BodyModel(bm_fname="/is/ps3/nsaini/projects/copenet/src/copenet/data/smplx/models/smplx/SMPLX_NEUTRAL.npz")

# bl_cmap = cm.get_cmap("Reds",30)
# ap_cmap = cm.get_cmap("Blues",30)
# app_cmap = cm.get_cmap("Greens",30)
# viridis = cm.get_cmap("viridis",10)

pl_verts0 = res["pl_verts0"]
# im0 = res["im0"]

import bpy
from bpy import context, data, ops
import math

ops.object.empty_add(type='PLAIN_AXES')
group = context.active_object



# camera_data = bpy.data.cameras.new(name='Camera')
# cam = bpy.data.objects.new('Camera', camera_data)
# bpy.context.scene.collection.objects.link(cam)
# cam.rotation_euler[0] = math.radians(180)
# cam.data.lens = 27
# cam.data.shift_x = -0.03
# cam.data.shift_y = 0.01

D = bpy.data
C = bpy.context
empty = D.objects.new("empty",None)
C.scene.collection.objects.link(empty)


ap_cmap = cm.get_cmap("Blues",8)
app_cmap = cm.get_cmap("Greens",8)

# bpy.context.scene.use_nodes = True
# tree = bpy.context.scene.node_tree

# # clear default nodes
# for node in tree.nodes:
#     tree.nodes.remove(node)

# # create input image node
# image_node = tree.nodes.new(type='CompositorNodeImage')
# image_node.location = 0,0

# render_node = tree.nodes.new('CompositorNodeRLayers')
# render_node.location = 400,0

# alphaover_node = tree.nodes.new('CompositorNodeAlphaOver')
# alphaover_node.location = 400,0

# # create output node
# comp_node = tree.nodes.new('CompositorNodeComposite')   
# comp_node.location = 800,0

# # link nodes0
# links = tree.links
# links.new(image_node.outputs[0], alphaover_node.inputs[1])
# links.new(render_node.outputs[0], alphaover_node.inputs[2])
# links.new(alphaover_node.outputs[0], comp_node.inputs[0])

# context.scene.render.film_transparent = True

# cmap_count = 10
# for i in range(2067,2230,10):
#     bl_smpl_mesh = bpy.data.meshes.new("bl_SMPL")
#     bl_smpl_obj = bpy.data.objects.new(bl_smpl_mesh.name,bl_smpl_mesh)
#     bl_smpl_mesh.from_pydata(bl_pred_vertices_cam0[i], [], list(smplx_model.f.detach().cpu().numpy()))
#     bpy.context.scene.collection.objects.link(bl_smpl_obj)
#     bl_mat = bpy.data.materials.new("bl_mat")
#     bl_mat.diffuse_color = (bl_cmap(cmap_count)[0],bl_cmap(cmap_count)[1],bl_cmap(cmap_count)[2],bl_cmap(cmap_count)[3])
#     bl_smpl_obj.active_material = bl_mat

#     ap_smpl_mesh = bpy.data.meshes.new("ap_SMPL")
#     ap_smpl_obj = bpy.data.objects.new(ap_smpl_mesh.name,ap_smpl_mesh)
#     ap_smpl_mesh.from_pydata(airpose_pred_vertices_cam0[i], [], list(smplx_model.f.detach().cpu().numpy()))
#     bpy.context.scene.collection.objects.link(ap_smpl_obj)
#     ap_mat = bpy.data.materials.new("ap_mat")
#     ap_mat.diffuse_color = (ap_cmap(cmap_count)[0],ap_cmap(cmap_count)[1],ap_cmap(cmap_count)[2],ap_cmap(cmap_count)[3])
#     ap_smpl_obj.active_material = ap_mat

#     smpl_mesh = bpy.data.meshes.new("SMPL")
#     smpl_obj = bpy.data.objects.new(smpl_mesh.name,smpl_mesh)
#     smpl_mesh.from_pydata(pl_verts0[i], [], list(smplx_model.f.detach().cpu().numpy()))
#     bpy.context.scene.collection.objects.link(smpl_obj)
#     mat = bpy.data.materials.new("mat")
#     mat.diffuse_color = (app_cmap(cmap_count)[0],app_cmap(cmap_count)[1],app_cmap(cmap_count)[2],app_cmap(cmap_count)[3])
#     smpl_obj.active_material = mat
#     smpl_obj.parent = group

#     cmap_count += 1

    # im = bpy.data.images.load(res_train["im0"][i])
    # image_node.image = im
    # cam.data.show_background_images = True
    # bg = cam.data.background_images.new()
    # bg.image = im

smpl_mesh = bpy.data.meshes.new("app_anim_SMPL")
smpl_obj = bpy.data.objects.new(smpl_mesh.name,smpl_mesh)
smpl_mesh.from_pydata(pl_verts0[0], [], list(smplx_model.f.detach().cpu().numpy()))
bpy.context.scene.collection.objects.link(smpl_obj)
mat = bpy.data.materials.new("mat")
mat.diffuse_color = (app_cmap(5)[0],app_cmap(5)[1],app_cmap(5)[2],app_cmap(5)[3])
smpl_obj.active_material = mat
smpl_obj.parent = empty

# smpl_mesh2 = bpy.data.meshes.new("app_anim_SMPL2")
# smpl_obj2 = bpy.data.objects.new(smpl_mesh2.name,smpl_mesh2)
# smpl_mesh2.from_pydata(airpose_pred_vertices_cam0[0], [], list(smplx_model.f.detach().cpu().numpy()))
# bpy.context.scene.collection.objects.link(smpl_obj2)
# mat2 = bpy.data.materials.new("mat")
# mat2.diffuse_color = (ap_cmap(5)[0],ap_cmap(5)[1],ap_cmap(5)[2],ap_cmap(5)[3])
# smpl_obj2.active_material = mat2
# smpl_obj2.parent = empty

# every frame change, this function is called.
def my_handler(scene):
    frame = scene.frame_current
    n = frame
    
    ob = bpy.data.objects.get("app_anim_SMPL")
    ob.data.clear_geometry()
    ob.data.from_pydata(pl_verts0[n],[], list(smplx_model.f.detach().cpu().numpy()))

    # ob2 = bpy.data.objects.get("app_anim_SMPL2")
    # ob2.data.clear_geometry()
    # ob2.data.from_pydata(airpose_pred_vertices_cam0[n],[], list(smplx_model.f.detach().cpu().numpy()))

    # im = bpy.data.images.load(im0[n])
    # image_node.image = im
    # cam.data.show_background_images = True
    # bg = cam.data.background_images.new()
    # bg.image = im
    # ob_cam1 = bpy.data.objects.get("cam1")
    # ob_cam1.location.x = cam1_wrt_origin[n].detach().cpu().numpy()[0,3]
    # ob_cam1.location.y = cam1_wrt_origin[n].detach().cpu().numpy()[1,3]
    # ob_cam1.location.z = cam1_wrt_origin[n].detach().cpu().numpy()[2,3]
    # cam1_obj.rotation_quaternion.w = cam1_wrt_origin_quat[0,0]
    # cam1_obj.rotation_quaternion.x = cam1_wrt_origin_quat[0,1]
    # cam1_obj.rotation_quaternion.y = cam1_wrt_origin_quat[0,2]
    # cam1_obj.rotation_quaternion.z = cam1_wrt_origin_quat[0,3]


bpy.app.handlers.frame_change_pre.append(my_handler)


bpy.ops.import_mesh.stl(filepath="/is/ps3/nsaini/projects/mcms/cam.stl")
cam_stl_obj = [obj for obj in bpy.context.scene.objects if obj.name.lower()=="cam"]
cam_stl_obj[0].name = "cam_0"
cam_obj = cam_stl_obj[0]
cam_obj.parent = empty
cam_obj.location[0] = 0
cam_obj.location[1] = 0
cam_obj.location[2] = 0
cam_obj.scale[0] = 10
cam_obj.scale[1] = 10
cam_obj.scale[2] = 10
# cam_obj.rotation_mode = 'QUATERNION'
# cam_obj.rotation_quaternion = cam_rots[dat_idx,cam,idx]
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


for idi,i in enumerate(np.arange(-5.5,5.5)):
    for idj,j in enumerate(np.arange(-5.5,5.5)):
        new_plane((i,j,0),0.95,"plane_{}_{}".format(idi,idj))

# set camera pose
render_cam = [obj for obj in bpy.context.scene.objects if obj.name=="Camera"][0]
render_cam.rotation_mode = 'XYZ'
render_cam.location.x = 15.6
render_cam.location.y = -39.6
render_cam.location.z = 24.3
render_cam.rotation_euler[0] = 55.9 * (math.pi/180.0)
render_cam.rotation_euler[1] = 0.707 * (math.pi/180.0)
render_cam.rotation_euler[2] = 1.7 * (math.pi/180.0)

# delete point light
point_light_obj = [obj for obj in bpy.context.scene.objects if obj.name=="Light"][0]
bpy.data.objects.remove(point_light_obj,do_unlink=True)

# add area light
light_data = bpy.data.lights.new(name="area_light", type='AREA')
light_data.energy = 5000
light_data.shape = 'RECTANGLE'
light_data.size = 30
light_data.size_y = 11

light_obj = bpy.data.objects.new(name="area_light", object_data=light_data)

bpy.context.collection.objects.link(light_obj)
light_obj.location.x = 15.26
light_obj.location.y = -2.57
light_obj.location.z = 6.55


bpy.ops.import_scene.fbx(filepath="/is/ps3/nsaini/projects/mcms/axis.fbx")