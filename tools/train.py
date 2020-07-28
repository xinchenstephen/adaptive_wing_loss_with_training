from __future__ import print_function, division
import torch
import argparse
import numpy as np
import torch.nn as nn
import time
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.evaler import eval_model
from core.trainer import train_model
from core.dataloader import get_dataset
from core import models
from tensorboardX import SummaryWriter
from utils.utils import create_logger
import torch.optim as optim
from core.ada_wing_loss import AWing,Loss_weighted


# Parse arguments
parser = argparse.ArgumentParser()
# Dataset paths
parser.add_argument('--image_root', type=str, default = "./data/images",
                    help='image directory')
parser.add_argument('--val_landmarks_data', type=str, required= True,
                    help='Validation landmarks file')
parser.add_argument('--train_landmarks_data', type=str, required= True,
                    help='train landmarks file')
parser.add_argument('--num_landmarks', type=int, default=68,
                    help='Number of landmarks')

# Checkpoint and pretrained weights
parser.add_argument('--ckpt_save_path', type=str,default = "./output/" + time.strftime('%Y-%m-%d-%H-%M'),
                    help='a directory to save checkpoint file')
parser.add_argument('--pretrained_weights', type=str, default= "None",
                    help='a directory to save pretrained_weights')

# Eval options
parser.add_argument('--batch_size', type=int, default=8,
                    help='learning rate decay after each epoch')

# Network parameters
parser.add_argument('--hg_blocks', type=int, default=4,
                    help='Number of HG blocks to stack')
parser.add_argument('--gray_scale', type=str, default="False",
                    help='Whether to convert RGB image into gray scale during training')
parser.add_argument('--end_relu', type=str, default="False",
                    help='Whether to add relu at the end of each HG module')

# Logger parameters
parser.add_argument('--tensorboard', type = str, default = "./log/" + time.strftime('%Y-%m-%d-%H-%M'),
                                                help = 'a directory to store the log file')

args = parser.parse_args()
IMG_ROOT = args.image_root
VAL_DATA = args.val_landmarks_data
TRAIN_DATA = args.train_landmarks_data
CKPT_SAVE_PATH = args.ckpt_save_path
BATCH_SIZE = args.batch_size
PRETRAINED_WEIGHTS = args.pretrained_weights
GRAY_SCALE = False if args.gray_scale == 'False' else True
HG_BLOCKS = args.hg_blocks
END_RELU = False if args.end_relu == 'False' else True
NUM_LANDMARKS = args.num_landmarks
TENSORBOARD_PATH = args.tensorboard


#other parameters
EPOCHES = 60
LR = 0.001
LR_STEP = [30,50]
LR_FACTOR = 0.1
GPU = list((0,))
SCALE_FACTOR = 0.22
ROT_FACTOR = 0
FLIP = True
data_para = {"scale_factor": SCALE_FACTOR, "rot_factor":ROT_FACTOR, "if_flip": FLIP}

#create the logger
logger, output_dir, tensorboard_dir = create_logger(CKPT_SAVE_PATH,TENSORBOARD_PATH)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


writer_dict = {
    'writer': SummaryWriter(log_dir="./log/" +  time.strftime('%Y-%m-%d-%H-%M')),
    'train_global_steps': 0,
    'valid_global_steps': 0,
}

train_dataloaders, train_dataset_sizes = get_dataset(IMG_ROOT, TRAIN_DATA,
                                         BATCH_SIZE, data_para, NUM_LANDMARKS, "train")

val_dataloaders, val_dataset_sizes = get_dataset(IMG_ROOT, VAL_DATA,
                                         BATCH_SIZE,  data_para, NUM_LANDMARKS, "val")
                                        
use_gpu = torch.cuda.is_available()
model_ft = models.FAN(HG_BLOCKS, END_RELU, GRAY_SCALE, NUM_LANDMARKS)


if PRETRAINED_WEIGHTS != "None":
    checkpoint = torch.load(PRETRAINED_WEIGHTS)
    if 'state_dict' not in checkpoint:
        model_ft.load_state_dict(checkpoint)
    else:
        pretrained_weights = checkpoint['state_dict']
        model_weights = model_ft.state_dict()
        pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                              if k in model_weights}
        model_weights.update(pretrained_weights)
        model_ft.load_state_dict(model_weights)

model_ft = nn.DataParallel(model_ft, device_ids=GPU).cuda()

#optimizer
optimizer =  optim.Adam(
                filter(lambda p: p.requires_grad, model_ft.parameters()),
                lr= LR
            )

#loss function
criterion =  Loss_weighted() #adpative_wing_loss


if isinstance(LR_STEP, list):
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, LR_STEP,
        LR_FACTOR, -1
    )
else:
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, LR_STEP,
        LR_FACTOR, -1
    )
    
model_ft = train_model(model_ft,train_dataloaders,train_dataset_sizes, val_dataloaders, val_dataset_sizes, BATCH_SIZE,optimizer,criterion,lr_scheduler,writer_dict, logger, epoches = EPOCHES,save_path=output_dir )


###储存最后的模型########
time_str = time.strftime('%Y-%m-%d-%H-%M')
final_model_state_file = os.path.join(output_dir,
                                        'final_state' + time_str + ".pth")
torch.save(model_ft.module.state_dict(), final_model_state_file)
writer_dict['writer'].close()

# model_ft = eval_model(model_ft, dataloaders, dataset_sizes, writer, use_gpu, 1, 'val', CKPT_SAVE_PATH, NUM_LANDMARKS)

