import os
import glob
import subprocess
import sys

# image dir index to process
idx = int(sys.argv[1]) + 480

# get list of image directories
imdirs = sorted(glob.glob("/ps/project/datasets/Human3.6M/images/*/*"))
outdir = "/is/cluster/nsaini/Datasets/Human3.6m/"


imdir = imdirs[idx]


# apose_outdir = os.path.join(outdir,"apose_res",*imdir.split("/")[-2:])
# os.makedirs(apose_outdir,exist_ok=True)


opose_outdir = os.path.join(outdir,"opose_res",*imdir.split("/")[-2:-1])
os.makedirs(opose_outdir,exist_ok=True)
opose_outfile = os.path.join(opose_outdir,*imdir.split("/")[-1:]) + ".pkl"

# subprocess.run(["singularity", "exec", "--nv", "-B", "/is:/is", "-B",
#                     "/home/nsaini/Datasets:/home/nsaini/Datasets",
#                     "/is/ps3/nsaini/projects/openpose_scripts/openpose.simg", "python3",
#                     "/AlphaPose/demo.py", 
#                     "--indir", imdir, "--outdir", apose_outdir, "--format", "open"])

subprocess.run(["singularity", "exec", "--nv", "-B", "/is:/is", "-B", 
                    "/home/nsaini/Datasets:/home/nsaini/Datasets",
                    "/is/ps3/nsaini/projects/openpose_scripts/openpose.simg",
                    "python3", "/is/ps3/nsaini/projects/savitr_pe/src/scripts/openpose_script_modified.py", 
                    "--input_dir", imdir, "--pkl_path", opose_outfile])