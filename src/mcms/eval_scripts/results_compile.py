import yaml
import glob
from os.path import join as ospj
import numpy as np
import os

results = {}
cpe = {}
coe = {}
smpl_moe = {}
smpl_mpe = {}
smpl_ra_mpjpe = {}
smpl_ra_mpvpe = {}
smpl_mpjpe = {}
cpe_data = {}
coe_data = {}
smpl_moe_data = {}
smpl_mpe_data = {}
smpl_ra_mpjpe_data = {}
smpl_ra_mpvpe_data = {}
smpl_mpjpe_data = {}
for fl in sorted(glob.glob("/is/ps3/nsaini/projects/mcms/mcms_logs/fittings/all_res_wth_sigma/*")):
    dat = yaml.safe_load(open(ospj(fl,"0000","gt_and_res_strt_off_10.yml"),"r"))
    results[fl.split("/")[-1]] = {"coe":str(round(dat["COE_mean"],2))+"+-"+str(round(dat["COE_std"],2)),
                                "cpe":str(round(100*dat["CPE_mean"],2))+"+-"+str(round(100*dat["CPE_std"],2)),
                                "smpl_moe":str(round(dat["SMPL_MOE"],2))+"+-"+str(round(dat["SMPL_MOE_std"],2)),
                                "smpl_mpe":str(round(100*dat["SMPL_MPE"],2))+"+-"+str(round(100*dat["SMPL_MPE_std"],2)),
                                "smpl_ra_mpjpe":str(round(100*dat["SMPL_MPJPE_tau0_phi0_beta0"],2))+"+-"+str(round(100*dat["SMPL_MPJPE_tau0_phi0_beta0_std"],2)),
                                "smpl_ra_mpvpe":str(round(100*dat["SMPL_MPVPE_tau0_phi0_beta0"],2))+"+-"+str(round(100*dat["SMPL_MPVPE_tau0_phi0_beta0_std"],2)),
                                "smpl_mpjpe":str(round(100*dat["SMPL_MPJPE"],2))+"+-"+str(round(100*dat["SMPL_MPJPE_std"],2))}
    
    coe[fl.split("/")[-1]] = {"mu":round(dat["COE_mean"],2),"std":round(dat["COE_std"],2)}
    cpe[fl.split("/")[-1]] = {"mu":round(100*dat["CPE_mean"],2),"std":round(100*dat["CPE_std"],2)}
    smpl_moe[fl.split("/")[-1]] = {"mu":round(dat["SMPL_MOE"],2),"std":round(dat["SMPL_MOE_std"],2)}
    smpl_mpe[fl.split("/")[-1]] = {"mu":round(100*dat["SMPL_MPE"],2),"std":round(100*dat["SMPL_MPE_std"],2)}
    smpl_ra_mpjpe[fl.split("/")[-1]] = {"mu":round(100*dat["SMPL_MPJPE_tau0_phi0_beta0"],2),"std":round(100*dat["SMPL_MPJPE_tau0_phi0_beta0_std"],2)}
    smpl_ra_mpvpe[fl.split("/")[-1]] = {"mu":round(100*dat["SMPL_MPVPE_tau0_phi0_beta0"],2),"std":round(100*dat["SMPL_MPVPE_tau0_phi0_beta0_std"],2)}
    smpl_mpjpe[fl.split("/")[-1]] = {"mu":round(100*dat["SMPL_MPJPE"],2),"std":round(100*dat["SMPL_MPJPE_std"],2)}


    if os.path.exists(ospj(fl,"0000","gt_and_humor_res_strt_off_10.yml")):
        dat = yaml.safe_load(open(ospj(fl,"0000","gt_and_humor_res_strt_off_10.yml"),"r"))
        results["humor"] = {"coe":str(round(dat["COE_mean"],2))+"+-"+str(round(dat["COE_std"],2)),
                                "cpe":str(round(100*dat["CPE_mean"],2))+"+-"+str(round(100*dat["CPE_std"],2)),
                                "smpl_moe":str(round(dat["SMPL_MOE"],2))+"+-"+str(round(dat["SMPL_MOE_std"],2)),
                                "smpl_mpe":str(round(100*dat["SMPL_MPE"],2))+"+-"+str(round(100*dat["SMPL_MPE_std"],2)),
                                "smpl_ra_mpjpe":str(round(100*dat["SMPL_MPJPE_tau0_phi0_beta0"],2))+"+-"+str(round(100*dat["SMPL_MPJPE_tau0_phi0_beta0_std"],2)),
                                "smpl_ra_mpvpe":str(round(100*dat["SMPL_MPVPE_tau0_phi0_beta0"],2))+"+-"+str(round(100*dat["SMPL_MPVPE_tau0_phi0_beta0_std"],2)),
                                "smpl_mpjpe":str(round(100*dat["SMPL_MPJPE"],2))+"+-"+str(round(100*dat["SMPL_MPJPE_std"],2))}



import matplotlib.pyplot as plt