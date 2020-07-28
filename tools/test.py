from __future__ import print_function, division
import torch
import argparse
import numpy as np
import torch.nn as nn
import time
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.tester import test_model
from core.dataloader import get_dataset
from core import models
import cv2 as cv
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from PIL import Image



def savelandmarks(img,imgpath,landmarks,nme,output_dir,model_file,h,center_x,center_y):
    nme = round(nme,4)
    picname = os.path.basename(imgpath)
    model_name = os.path.basename(model_file)
    imgh,imgw = img.shape[0],img.shape[1]
    cv.rectangle(img,(center_x - int(h/2),center_y - int(h/2)),(center_x + int(h/2),center_y + int(h/2)),200)
    # cv.putText(img,str(nme),(int(imgh/9),int(imgw/9)),cv.FONT_HERSHEY_PLAIN,int(imgh/180),thickness = 5,color = (0,255,0))
    plt.figure()
    plt.imshow(img)
    plt.scatter(landmarks[:,0],landmarks[:,1],s = 1, c = "r")
    plt.text(int(imgh/9),int(imgw/9), str(nme),color = "red",fontsize = 8)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir,"_" + model_name + "_" + picname))
    plt.show()
    plt.close()

# Parse arguments
parser = argparse.ArgumentParser()
# Dataset paths
parser.add_argument('--image_root', type=str, default = "./data/images",
                    help='image directory')
parser.add_argument('--test_landmarks_data', type=str, required= True,
                    help='test landmarks file')
parser.add_argument('--num_landmarks', type=int, default=68,
                    help='Number of landmarks')

#model-file
parser.add_argument('--model-file', type=str, required = True,
                    help='model-file-name')


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


args = parser.parse_args()

TEST_IMG_ROOT = args.image_root
TEST_DATA_PATH = args.test_landmarks_data
BATCH_SIZE = args.batch_size
MODEL_FILE = args.model_file
GRAY_SCALE = False if args.gray_scale == 'False' else True
HG_BLOCKS = args.hg_blocks
END_RELU = False if args.end_relu == 'False' else True
NUM_LANDMARKS = args.num_landmarks

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



dataloaders, dataset_sizes = get_dataset(TEST_IMG_ROOT, TEST_DATA_PATH,
                                         BATCH_SIZE, NUM_LANDMARKS, type = "test")
use_gpu = torch.cuda.is_available()
model_ft = models.FAN(HG_BLOCKS, END_RELU, GRAY_SCALE, NUM_LANDMARKS)


if MODEL_FILE != "None":
    checkpoint = torch.load(MODEL_FILE)
    if 'state_dict' not in checkpoint:
        model_ft.load_state_dict(checkpoint)
    else:
        pretrained_weights = checkpoint['state_dict']
        # model_weights = model_ft.state_dict()
        # pretrained_weights = {k: v for k, v in pretrained_weights.items() \
        #                       if k in model_weights}
        # model_weights.update(pretrained_weights)
        model_ft.module.load_state_dict(pretrained_weights)

model_ft = model_ft.to(device)

global_nme,predictions,nmes = test_model(model_ft, dataloaders, dataset_sizes, use_gpu, 'test', NUM_LANDMARKS)
output_dir = "./output/300W/result/" +  time.strftime('%Y-%m-%d')
output_dir_obj = Path(output_dir)
if not output_dir_obj.exists():
    output_dir_obj.mkdir(parents = True, exist_ok = True)
data = pd.read_csv(TEST_DATA_PATH)

#see the landmarks
for i in range(len(predictions)):    # # print(state_dict)
        nme = nmes[i].item()
        imgcompath = data.iloc[i,0]
        h = int(data.iloc[i,1]*200)
        center_x = int(data.iloc[i,2])
        center_y = int(data.iloc[i,3])
        imgpath = os.path.join(TEST_IMG_ROOT, imgcompath)
        img = np.array(Image.open(imgpath).convert('RGB'))
        savelandmarks(img,imgpath,predictions[i],nme,output_dir,args.model_file,h,center_x,center_y)
