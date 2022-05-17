import os
import glob
from tqdm import tqdm
import shutil
from mcms.dsets import rich
import json
import torch

parent_dir_src = "/ps/project/datasets/AirCap_ICCV19/RICH_IPMAN/test/2021-06-15_Multi_IOI_ID_00186_Yoga1"
parent_dir_dest = "/ps/project/datasets/AirCap_ICCV19/RICH_tryout/test/2021-06-15_Multi_IOI_ID_00186_Yoga1"
rich_ds = rich.rich(parent_dir_src)

for cam in tqdm(range(7)):
    
    os.makedirs(os.path.join(parent_dir_dest,"forhumor",str(cam),"rgb_preprocess","raw_frames"),exist_ok=True)
    os.makedirs(os.path.join(parent_dir_dest,"forhumor",str(cam),"rgb_preprocess","op_keypoints"),exist_ok=True)

    for i,f in enumerate(tqdm(range(5,599))):
        dat = rich_ds.__getitem__(f,1)
        shutil.copy(dat["full_im_paths"][cam][0],
                    os.path.join(parent_dir_dest,"forhumor",str(cam),"rgb_preprocess","raw_frames","{:06d}.png".format(i)),
                    follow_symlinks=False)
        json.dump({"people":[{"pose_keypoints_2d": dat["j2d_op"][cam,0].detach().cpu().numpy().tolist()}]},
                open(os.path.join(parent_dir_dest,"forhumor",str(cam),"rgb_preprocess","op_keypoints","{:06d}.json".format(i)),"w"))

# dump camera intrinsics
cam_intrs = rich_ds.__getitem__(5,1)["cam_intr"]
for cam in tqdm(range(7)):
    with open(os.path.join(parent_dir_dest,"forhumor",str(cam),"rgb_preprocess","cam_intrinsics.json"),"w") as f:
        json.dump(cam_intrs[cam].tolist(),f)