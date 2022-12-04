# SmartMocap
![](./resources/teaser.jpg)

## Install

### Get apptainer
Install apptainer from [here](https://github.com/apptainer/apptainer/blob/main/INSTALL.md)

### Clone SmartMocap
```
git clone https://github.com/robot-perception-group/SmartMocap.git
cd SmartMocap
```

### Prepare the container
```
apptainer build --sandbox ./smartmocap_apptainer apptainer.def
apptainer shell --nv ./smartmocap_apptainer
poetry install
```

### Download data
Download the dataset you want to use and extract the tar file.
- [SmartMocap data](https://download.is.tue.mpg.de/download.php?domain=smartmocap&resume=1&sfile=VolleyDay.tar.gz)
- [AirPose data](https://download.is.tue.mpg.de/download.php?domain=smartmocap&resume=1&sfile=copenet_data.tar.gz)


### Download the pretrained MOP model
Download the MOP pretrained checkpoint file [here](https://download.is.tue.mpg.de/download.php?domain=smartmocap&resume=1&sfile=Smartmocap_mop_ckpt_release.tar.gz)

### Download SMPL model
Download SMPL models from [https://smpl.is.tue.mpg.de/](https://smpl.is.tue.mpg.de/)

### Download vposer model v2.0
Download VPoser model v2.0 from [https://smpl-x.is.tue.mpg.de/](https://smpl-x.is.tue.mpg.de/)

## Fitting config file
All the hyperparameters and paths need to be set in the file `src/mcms/fitting_scripts/fit_config.yml`. Descriptions are in the file itself.
For quicker execution on a particular dataset, we provide config for each dataset. You still need to set the paths (e.g. dataset path) in these files.
- SmartMocap data: `src/mcms/fitting_scripts/smartmocap_config.yml`
- RICH data: `src/mcms/fitting_scripts/rich_config.yml`
- Airpose data: `src/mcms/fitting_scripts/airpose_config.yml`
Replace the content of `src/mcms/fitting_scripts/fit_config.yml` with the content of any desired dataset config file to run on that dataset.


## Fitting

```
apptainer shell --nv ./smartmocap_apptainer
. .venv/bin/activate
python src/mcms/fitting_scripts/fitting_in_vp_latent_multires.py name_of_the_trial
```
Results will be in the folder `Smartmocap_logs/fittings/name_of_the_trial`.

## Visualization
Coming soon