import matplotlib
import math
import torch
import copy
import time
from torch.autograd import Variable
import shutil
from skimage import io
import numpy as np
from utils.utils import fan_NME, show_landmarks, get_preds_fromhm
from PIL import Image, ImageDraw
import os
import sys
import cv2
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_model(model, dataloaders, dataset_sizes,
               use_gpu=True, dataset='test',
               save_path='./', num_landmarks=68):

    model.eval()
    running_loss = 0
    global_nme = 0
    step = 0
    total_nme = 0
    total_count = 0
    fail_count_010 = 0
    fail_count_008 = 0
    preds = np.zeros((dataset_sizes[dataset],num_landmarks,2))
    nmes = np.zeros((dataset_sizes[dataset], ))
    # running_corrects = 0

    # Iterate over data.
    with torch.no_grad():
        for data in dataloaders[dataset]:
            total_runtime = 0
            run_count = 0
            step_start = time.time()
            step += 1
            # get the inputs
            inputs = data['image'].type(torch.FloatTensor)
            labels_heatmap = data['heatmap'].type(torch.FloatTensor)
            labels_boundary = data['boundary'].type(torch.FloatTensor)
            landmarks = data['landmarks'].type(torch.FloatTensor)
            loss_weight_map = data['weight_map'].type(torch.FloatTensor)
            # wrap them in Variable
            if use_gpu:
                inputs = inputs.to(device)
                labels_heatmap = labels_heatmap.to(device)
                labels_boundary = labels_boundary.to(device)
                loss_weight_map = loss_weight_map.to(device)
            else:
                inputs, labels_heatmap = Variable(inputs), Variable(labels_heatmap)
                labels_boundary = Variable(labels_boundary)
            labels = torch.cat((labels_heatmap, labels_boundary), 1)
            single_start = time.time()
            outputs, boundary_channels = model(inputs)

            single_end = time.time()
            total_runtime += single_end - single_start
            run_count += 1
            step_end = time.time()
            for i in range(inputs.shape[0]):
                img = inputs[i]
                img = img.cpu().numpy()
                img = img.transpose((1, 2, 0))*255.0
                img = img.astype(np.uint8)
                # img = Image.fromarray(img)
                # pred_heatmap = outputs[-1][i].detach().cpu()[:-1, :, :] detach()和.data作用基本相同 都是防止反传改变梯度
                center = data['center'][i].numpy()
                scale = data['scale'][i]
                pred_heatmap = outputs[-1][:, :-1, :, :][i].detach().cpu()
                pred_landmarks, preds_ori = get_preds_fromhm(pred_heatmap.unsqueeze(0), center = center, scale = scale)
                pred_landmarks = pred_landmarks.squeeze().numpy()
                preds[data['index'][i]] = preds_ori

                gt_landmarks = data['landmarks'][i].numpy()
                if num_landmarks == 68:
                    # left_eye = np.average(gt_landmarks[36:42], axis=0)
                    # right_eye = np.average(gt_landmarks[42:48], axis=0)
                    # norm_factor = np.linalg.norm(left_eye - right_eye)
                    # print("change factor")
                    norm_factor = np.linalg.norm(gt_landmarks[36]- gt_landmarks[45])
                elif num_landmarks == 98:
                    norm_factor = np.linalg.norm(gt_landmarks[60]- gt_landmarks[72])
                elif num_landmarks == 19:
                    left, top = gt_landmarks[-2, :]
                    right, bottom = gt_landmarks[-1, :]
                    norm_factor = math.sqrt(abs(right - left)*abs(top-bottom))
                    gt_landmarks = gt_landmarks[:-2, :]
                elif num_landmarks == 29:
                    # norm_factor = np.linalg.norm(gt_landmarks[8]- gt_landmarks[9])
                    norm_factor = np.linalg.norm(gt_landmarks[16]- gt_landmarks[17])

                # show_landmarks(img,pred_heatmap.numpy(),gt_landmarks,labels_heatmap[i].detach().cpu().numpy())

                single_nme = (np.sum(np.linalg.norm(pred_landmarks*4 - gt_landmarks, axis=1)) / pred_landmarks.shape[0]) / norm_factor

                nmes[data['index'][i]] = single_nme
                total_count += 1
                if single_nme > 0.08:
                    fail_count_008 += 1
                if single_nme > 0.10:
                    fail_count_010 += 1

            # gt_landmarks = landmarks.numpy()
            # pred_heatmap = outputs[-1].to('cpu').numpy()
            gt_landmarks = landmarks
            batch_nme = fan_NME(outputs[-1][:, :-1, :, :].detach().cpu(), gt_landmarks, num_landmarks)
            # batch_nme = 0
            total_nme += batch_nme
    

        global_nme = total_nme / dataset_sizes['test']
        print('val_NME: {:.6f} Failure( > 0.08) Rate: {:.6f} Failure( > 0.10) Rate: {:.6f} Total Count: {:.6f} '.format(global_nme, fail_count_008/total_count, fail_count_010/total_count, total_count))

    print('Etestuation done! Average NME: {:.6f}'.format(global_nme))
    print('Everage runtime for a single batch: {:.6f}'.format(total_runtime/run_count))
    return global_nme, preds,nmes
