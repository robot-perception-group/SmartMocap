##   This python script launches openpose to get 2D joints for all images under an input directory  ##
##   Author: Jinlong Yang
##   Email:  jinlong.yang@tuebingen.mpg.de

import os
import sys
from os.path import join, isdir, isfile
#from tqdm import tqdm
import argparse
import cv2
import numpy as np
import pickle as pkl
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--im_paths', type=str, default=None)
parser.add_argument('--outdir', type=str, default=None)

def openpose_ini(openposepy_dir = '/openpose/build/python/'):
    try:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(openposepy_dir);
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = join(openposepy_dir,"../../models/")
    params["hand"] = True
    params["face"] = True

    # # Add others in path?
    # for i in range(0, len(args[1])):
    #     curr_item = args[1][i]
    #     if i != len(args[1])-1: next_item = args[1][i+1]
    #     else: next_item = "1"
    #     if "--" in curr_item and "--" in next_item:
    #         key = curr_item.replace('-','')
    #         if key not in params:  params[key] = "1"
    #     elif "--" in curr_item and "--" not in next_item:
    #         key = curr_item.replace('-','')
    #         if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    datum = op.Datum()

    return opWrapper, datum


def openpose_run(opWrapper, datum, image_file):
    # Process Image
    imageToProcess = cv2.imread(image_file)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])

    return datum

    
import json
def export_json(datum, file_name):
    dic = {}
    dic['version'] = '1.5'
    dic["people"] = []
    person_dic = {}
    if hasattr(datum, 'poseKeypoints'):
        if datum.poseKeypoints.shape:
            for i in range(0, datum.poseKeypoints.shape[0]):
                person_dic["person_id"] = [-1]
                person_dic["pose_keypoints_2d"] = (datum.poseKeypoints[0,:,:].reshape((-1))).tolist()
        else:
            person_dic["person_id"] = []
            person_dic["pose_keypoints_2d"] = []
            person_dic["pose_keypoints_2d"] = []

    if hasattr(datum, 'faceKeypoints'):
        if datum.faceKeypoints.shape:
            for i in range(0, datum.faceKeypoints.shape[0]):
                person_dic["face_keypoints_2d"] = (datum.faceKeypoints[0,:,:].reshape((-1))).tolist()
        else:
            person_dic["face_keypoints_2d"] = []

    if hasattr(datum, 'handKeypoints'):
        if datum.handKeypoints[0].shape:
            for i in range(0, datum.handKeypoints[0].shape[0]):
                person_dic["hand_left_keypoints_2d"] = (datum.handKeypoints[0][0,:,:].reshape((-1))).tolist()
        else:
            person_dic["hand_left_keypoints_2d"] = []

        if datum.handKeypoints[1].shape:
            for i in range(0, datum.handKeypoints[1].shape[0]):
                person_dic["hand_right_keypoints_2d"] = (datum.handKeypoints[1][0,:,:].reshape((-1))).tolist()
        else:
            person_dic["hand_right_keypoints_2d"] = []
              

    dic["people"].append(person_dic)

    with open(file_name, 'w') as fp:
        json.dump(dic, fp)
    return

def process_all_images_in_directory(im_paths): # input_dir contains .png or jpg images
    # Initialize  openpose wrapperh:
    opWrapper, datum = openpose_ini()

    # Get all images under current directory
    images = sorted(glob.glob(im_paths))

    print(len(images), ' images in total will be processed.')
    
    kp_dict = {}
    for image in images:
        print("Processing ", image)
        datum = openpose_run(opWrapper, datum, image)
        poseKeypoints_2d = datum.poseKeypoints
        faceKeypoints_2d = datum.faceKeypoints
        handKeypoints_2d_left = datum.handKeypoints[0]
        handKeypoints_2d_right = datum.handKeypoints[1]

        freecam_outdir = os.path.join("/".join(image.split("/")[:-2]),"freecam_keypoints")
        os.makedirs(freecam_outdir, exist_ok=True)

        json_file_name = os.path.join(freecam_outdir, image.split("/")[-1][:-4]+'_keypoints.json')
        export_json(datum, json_file_name)

        kp_dict[image[:-4]] = {"pose": poseKeypoints_2d,
                                "face": faceKeypoints_2d,
                                "hand_left": handKeypoints_2d_left,
                                "hand_right": handKeypoints_2d_right}

        # np.save(join(output_dir, image[:-4]+'_2djoint.npy'), poseKeypoints_2d)
        cv2.imwrite(os.path.join(freecam_outdir, image.split("/")[-1][:-4]+'_openpose.png'), datum.cvOutputData)
    
    # pkl.dump(kp_dict, open(pkl_path,"wb"))
    
if __name__ == '__main__':
    args = parser.parse_args()
    process_all_images_in_directory(args.im_paths)
