import os
os.environ["PYOPENGL_PLATFORM"] = 'egl'
from mcms.models import mcms
import yaml

ckpt_path = "/is/ps3/nsaini/projects/mcms/mcms_logs/test/v000_3/checkpoints/epoch=149-step=4499.ckpt"
hparams = yaml.safe_load(open("/".join(ckpt_path.split("/")[:-2])+"/hparams.yaml","r"))

net = mcms.mcms.load_from_checkpoint(checkpoint_path=ckpt_path, hparams=hparams)
net.eval()

from mcms.dsets import h36m
from torch.utils.data import DataLoader
ds = h36m.h36m(hparams)
dl = DataLoader(ds,batch_size=1)

out,_,_ = net.fwd_pass_and_loss(next(iter(dl)))

import numpy as np
np.save(".".join(ckpt_path.split(".")[:-1]),out["smpl_out_v"])
