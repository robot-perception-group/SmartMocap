"""
This file runs the main training/val loop, etc... using Lightning Trainer    
"""
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
import itertools
import os.path as osp
import os, sys, time
import pathlib
from mcms.models import mcms
os.environ["PYOPENGL_PLATFORM"] = 'egl'
# os.environ['EGL_DEVICE_ID'] = os.environ['GPU_DEVICE_ORDINAL'].split(',')[0]

# sets seeds for numpy, torch, etc...
# must do for DDP to work well

import sys
idx_num = sys.argv[1]

hparams = yaml.safe_load(open(osp.join(pathlib.Path(__file__).parent.resolve(),"hparams.yaml")))

search_params_keys = [x for x in hparams if type(hparams[x]) is list]
search_params_vals = list(itertools.product(*[hparams[x] for x in hparams if type(hparams[x]) is list]))

for i,k in enumerate(search_params_keys):
    hparams[k] = search_params_vals[idx_num][i]


version = "v{:03d}_".format(int(idx_num)) + ",".join([str(search_params_keys[i])+"="+str(search_params_vals[idx_num][i]) for i in range(len(search_params_keys))])



if osp.exists(osp.join(hparams["train_logdir"], hparams["train_name"], version)):

    print("##########################")
    print("resuming from "+osp.join(hparams["train_logdir"],hparams["train_name"], version))
    print("##########################")

    hparams = yaml.safe_load(open(osp.join(hparams["train_logdir"],hparams["train_name"],version,"hparams.yaml")))

    seed_everything(seed=hparams["train_PL_GLOBAL_SEED"])

    model = mcms.mcms(hparams=hparams)

    logger = TensorBoardLogger(save_dir=hparams["train_logdir"], name=hparams["train_name"],version=version)

    if osp.exists(osp.join(hparams["train_logdir"], hparams["train_name"],version ,"checkpoints","last.ckpt")):
        trainer = Trainer(max_epochs=hparams["train_max_epochs"],
                                        gpus=hparams["train_num_gpus"],
                                        resume_from_checkpoint=osp.join(hparams["train_logdir"], hparams["train_name"],version ,"checkpoints","last.ckpt"),
                                        logger=logger,
                                        check_val_every_n_epoch=hparams["train_check_val_every_n_epoch"],
                                        callbacks=[ModelCheckpoint(monitor="val_loss",mode="min",save_top_k=1,save_last=True)])
    else:
        trainer = Trainer(max_epochs=hparams["train_max_epochs"],
                                        gpus=hparams["train_num_gpus"],
                                        logger=logger,
                                        check_val_every_n_epoch=hparams["train_check_val_every_n_epoch"],
                                        callbacks=[ModelCheckpoint(monitor="val_loss",mode="min",save_top_k=1,save_last=True)])

else:
    
    if hparams["train_PL_GLOBAL_SEED"] is None:
        seed_everything(123)
        hparams["train_PL_GLOBAL_SEED"] = int(os.environ["PL_GLOBAL_SEED"])
    else:
        seed_everything(int(hparams["train_PL_GLOBAL_SEED"]))

    if hparams["train_pretrained_checkpoint"] == None:
        model = mcms.mcms(hparams)
    else:
        print("Loading from pretrained checkpoint " + hparams["train_pretrained_checkpoint"])
        model = mcms.mcms.load_from_checkpoint(checkpoint_path=hparams["train_pretrained_checkpoint"])

    logger = TensorBoardLogger(save_dir=hparams["train_logdir"], name=hparams["train_name"],version=version)
    
    os.makedirs(osp.join(hparams["train_logdir"],hparams["train_name"],version))

    yaml.dump(hparams, open(osp.join(hparams["train_logdir"],hparams["train_name"],version,"hparams.yaml"),"w"))
    
    trainer = Trainer(max_epochs=hparams["train_max_epochs"],
                                        gpus=hparams["train_num_gpus"],
                                        logger=logger,
                                        check_val_every_n_epoch=hparams["train_check_val_every_n_epoch"],
                                        callbacks=[ModelCheckpoint(monitor="val_loss",mode="min",save_top_k=1,save_last=True)])

trainer.fit(model)


