from statistics import median
import cv2
import numpy as np
import pickle as pkl
from tqdm import tqdm
import glob

data = pkl.load(open("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/volley_data_teaser4/0000/stage_06/_seq_start_01450.pkl","rb"))

projected_ims_paths = sorted(glob.glob("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/volley_data_teaser4/0000/stage_06/_seq_start_01450/cam_03/*.png"))
projected_ims = np.stack([cv2.imread(x)[:,:,::-1]/255. for x in projected_ims_paths[::10]])
images = np.stack([cv2.imread(data["full_im_paths"][3][x])[:,:,::-1]/255. for x in range(0,len(data["full_im_paths"][3]),10)])

median_img = np.median(images, axis=0)

res_img = median_img.copy()

threshold = 0.30
for x in tqdm(range(66,76)):
    im = images[x]
    mask1 = (np.abs(im - median_img)**2).sum(-1)
    mask1[mask1 > threshold] = 1
    mask1[mask1 < threshold] = 0
    mask1 = cv2.dilate(mask1,cv2.getStructuringElement(1,(5,5)))[:,:,np.newaxis]
    mask2 = (np.abs(projected_ims[x] - im)**2).sum(-1)
    mask2[mask2 > 0.1] = 1
    mask2[mask2 < 0.1] = 0
    mask2 = cv2.dilate(mask2,cv2.getStructuringElement(1,(5,5)))[:,:,np.newaxis]
    mask = mask1 + mask2
    mask[mask > 0.1] = 1
    mask[mask < 0.1] = 0
    res_img = (mask*im + (1-mask)*res_img)
    # import ipdb;ipdb.set_trace()