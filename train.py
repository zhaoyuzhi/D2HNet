import argparse
import os
import torch
import yaml
from easydict import EasyDict as edict

from trainer import Trainer, TwoPhaseTrainer

def attatch_to_config(args, opt):
    opt.num_gpus = args.num_gpus
    opt.save_path = args.save_path
    opt.log_path = args.log_path

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type = str, default = 'options/tp_deblurnet_v2_002.yaml', help = 'Path to option YAML file.')
    parser.add_argument('--num_gpus', type = int, default = 1, help = 'GPU numbers, 0 means cpu is used.')
    parser.add_argument('--save_path', type = str, default = 'snapshot/tp_deblurnet_v2_002', help = 'Path to save model.')
    parser.add_argument('--log_path', type = str, default = 'log_pt/tp_deblurnet_v2_002', help = 'Path to write log.')
    args = parser.parse_args()

    with open(args.opt, mode = 'r') as f:
        opt = edict(yaml.load(f))

    attatch_to_config(args, opt)

    print(opt)
    
    # ----------------------------------------
    #                  Train
    # ----------------------------------------
    
    trainer = TwoPhaseTrainer(opt = opt, num_gpus = opt.num_gpus)
    trainer.train()
    trainer.finish()
