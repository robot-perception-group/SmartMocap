
for f in /ps/project/datasets/AirCap_ICCV19/Human3.6m/images/S8/*; do
python scripts/demo.py --image_folder $f --output_folder /home/nsaini/Human3.6m/pare_res/S8 --no_render --mode folder
done