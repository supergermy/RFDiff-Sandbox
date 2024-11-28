#!/usr/bin/env python
"""
Inference script.

To run with base.yaml as the config,

> python run_inference.py

To specify a different config,

> python run_inference.py --config-name symmetry

where symmetry can be the filename of any other config (without .yaml extension)
See https://hydra.cc/docs/advanced/hydra-command-line-flags/ for more options.

"""

import re
import os, time, pickle, sys
import torch
from omegaconf import OmegaConf
import hydra
import logging
from rfdiffusion.util import writepdb_multi, writepdb
from rfdiffusion.inference import utils as iu
from hydra.core.hydra_config import HydraConfig
import numpy as np
import random
import glob


def xyz_to_pdb(coords, output_pdb):
    with open(output_pdb, 'w') as f:
        for i,xyz in enumerate(coords):
            f.write(f"ATOM{i+1:7d}  N   GLY A{i+1:4d}    {xyz[0][0].item():8.3f}{xyz[0][1].item():8.3f}{xyz[0][2].item():8.3f}  1.00  0.00           N\n")
            f.write(f"ATOM{i+1:7d}  CA  GLY A{i+1:4d}    {xyz[1][0].item():8.3f}{xyz[1][1].item():8.3f}{xyz[1][2].item():8.3f}  1.00  0.00           C\n")
            f.write(f"ATOM{i+1:7d}  C   GLY A{i+1:4d}    {xyz[2][0].item():8.3f}{xyz[2][1].item():8.3f}{xyz[2][2].item():8.3f}  1.00  0.00           C\n")
            f.write(f"ATOM{i+1:7d}  O   GLY A{i+1:4d}    {xyz[3][0].item():8.3f}{xyz[3][1].item():8.3f}{xyz[3][2].item():8.3f}  1.00  0.00           O\n")

def make_deterministic(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

@hydra.main(version_base=None, config_path="../config/inference", config_name="oneshot")
def main(conf: HydraConfig) -> None:
    log = logging.getLogger(__name__)
    if conf.inference.deterministic:
        make_deterministic()

    # Check for available GPU and print result of check
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        log.info(f"Found GPU with device_name {device_name}. Will run RFdiffusion on {device_name}")
    else:
        log.info("////////////////////////////////////////////////")
        log.info("///// NO GPU DETECTED! Falling back to CPU /////")
        log.info("////////////////////////////////////////////////")


    # Initialize sampler and target/contig.
    sampler = iu.sampler_selector(conf)
    shot_num = sampler.inf_conf.shot_num

    for _ in range(shot_num):
        # Loop over number of designs to sample.
        design_startnum = sampler.inf_conf.design_startnum
        if sampler.inf_conf.design_startnum == -1:
            existing = glob.glob(sampler.inf_conf.output_prefix + "*.pdb")
            indices = [-1]
            for e in existing:
                print(e)
                m = re.match(".*_(\d+)\.pdb$", e)
                print(m)
                if not m:
                    continue
                m = m.groups()[0]
                indices.append(int(m))
            design_startnum = max(indices) + 1

        for i_des in range(design_startnum, design_startnum + sampler.inf_conf.num_designs):
            if conf.inference.deterministic:
                make_deterministic(i_des)

            start_time = time.time()
            out_prefix = f"{sampler.inf_conf.output_prefix}_{i_des}"
            log.info(f"Making design {out_prefix}")
            if sampler.inf_conf.cautious and os.path.exists(out_prefix + ".pdb"):
                log.info(
                    f"(cautious mode) Skipping this design because {out_prefix}.pdb already exists."
                )
                continue

            x_init, seq_init = sampler.sample_init()

            x_t = torch.clone(x_init)
            seq_t = torch.clone(seq_init)
            t = sampler.inf_conf.final_step - 1

            msa_masked, msa_full, seq_in, xt_in, idx_pdb, t1d, t2d, xyz_t, alpha_t = sampler._preprocess(seq_init, x_t, t)

            N,L = msa_masked.shape[:2]


            if sampler.symmetry is not None:
                idx_pdb, sampler.chain_idx = sampler.symmetry.res_idx_procesing(res_idx=idx_pdb)

            msa_prev = None
            pair_prev = None
            state_prev = None

            with torch.no_grad():
                msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = sampler.model(msa_masked,
                                    msa_full,
                                    seq_in,
                                    xt_in,
                                    idx_pdb,
                                    t1d=t1d,
                                    t2d=t2d,
                                    xyz_t=xyz_t,
                                    alpha_t=alpha_t,
                                    msa_prev = msa_prev,
                                    pair_prev = pair_prev,
                                    state_prev = state_prev,
                                    t=torch.tensor(t),
                                    return_infer=True,
                                    motif_mask=sampler.diffusion_mask.squeeze().to(sampler.device))

            # prediction of X0 
            _, px0  = sampler.allatom(torch.argmax(seq_in, dim=-1), px0, alpha)
            px0    = px0.squeeze()[:,:14]

            xyz_to_pdb(px0, sampler.inf_conf.output_pdb)
            sampler.inf_conf.input_pdb = sampler.inf_conf.output_pdb


if __name__ == "__main__":
    main()
