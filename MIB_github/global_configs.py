import os
import torch

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["WANDB_PROGRAM"] = "main_mib.py"

DEVICE = torch.device("cuda:0")

# MOSI SETTING
ACOUSTIC_DIM = 74
VISUAL_DIM = 47
TEXT_DIM = 768
'''
# MOSEI SETTING
ACOUSTIC_DIM = 74
VISUAL_DIM = 35
TEXT_DIM = 768
'''
