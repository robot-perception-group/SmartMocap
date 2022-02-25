import os
os.environ["PYOPENGL_PLATFORM"] = 'egl'
from mcms.models import mcms
import yaml

ckpt_path = "/is/ps3/nsaini/projects/mcms/mcms_logs/test/v000_5/checkpoints/epoch=179-step=5399.ckpt"
hparams = yaml.safe_load(open("/".join(ckpt_path.split("/")[:-2])+"/hparams.yaml","r"))

from mcms.dsets import h36m
from torch.utils.data import DataLoader
ds = h36m.h36m(hparams)
dl = DataLoader(ds,batch_size=1)

