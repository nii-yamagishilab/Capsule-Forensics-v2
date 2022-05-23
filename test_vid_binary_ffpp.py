"""
Copyright (c) 2019, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------
Script for testing Capsule-Forensics-v2 on videos level by aggregating the predicted probabilities of their frames using FaceForensics++ database (Real, DeepFakes, Face2Face, FaceSwap)
"""

import sys
sys.setrecursionlimit(15000)
import os
import torch
import numpy as np
from torch.autograd import Variable
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from sklearn import metrics
from PIL import Image
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import math
import model_big


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default ='databases/faceforensicspp/test', help='path to test dataset')
parser.add_argument('--real', default ='0_original', help='real folder name')
parser.add_argument('--deepfakes', default ='1_deepfakes', help='fake folder name')
parser.add_argument('--face2face', default ='2_face2face', help='fake folder name')
parser.add_argument('--faceswap', default ='3_faceswap', help='fake folder name')
parser.add_argument('--batchSize', type=int, default=10, help='batch size')
parser.add_argument('--imageSize', type=int, default=300, help='the height / width of the input image to network')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--outf', default='checkpoints/binary_faceforensicspp', help='folder to output images and model checkpoints')
parser.add_argument('--random_sample', type=int, default=0, help='number of random sample to test')
parser.add_argument('--random', action='store_true', default=False, help='enable randomness for routing matrix')
parser.add_argument('--id', type=int, default=21, help='checkpoint ID')

opt = parser.parse_args()
print(opt)

img_ext_lst = ('jpg', 'jpeg', 'bmp', 'png', 'ppm', 'gif', 'tiff')

def get_file_list(path, ext_lst):
    file_lst = []

    for f in os.listdir(path):
        if os.path.isfile(os.path.join(path, f)):
            if f.lower().endswith(ext_lst):
                file_lst.append(f)

    return file_lst

def extract_file_name_without_count(in_str, sep_char='_'):
    n = len(in_str)
    pos = 0
    for i in range(n):
        if in_str[i] == sep_char:
            pos = i

    return in_str[0:pos]

def process_file_list(file_lst, sep_char='_'):
    result_lst = []

    for i in range(len(file_lst)):
        #remove extension
        filename = os.path.splitext(file_lst[i])[0]

        filename = extract_file_name_without_count(filename, sep_char)
        result_lst.append(filename)

    return result_lst

def classify_batch(vgg_ext, model, batch):
    n_sub_imgs = len(batch)

    if (opt.random_sample > 0):
        if n_sub_imgs > opt.random_sample:
            np.random.shuffle(batch)
            n_sub_imgs = opt.random_sample

        img_tmp = torch.FloatTensor([]).view(0, 3, opt.imageSize, opt.imageSize)

        for i in range(n_sub_imgs):
            img_tmp = torch.cat((img_tmp, batch[i]), dim=0)

        if opt.gpu_id >= 0:
            img_tmp = img_tmp.cuda(opt.gpu_id)

        input_v = Variable(img_tmp, requires_grad = False)

        x = vgg_ext(input_v)
        classes, class_ = model(x, random=opt.random)
        output_pred = class_.data.cpu().numpy()

    else:
        batchSize = opt.batchSize
        steps = int(math.ceil(n_sub_imgs*1.0/batchSize))

        output_pred = np.array([], dtype=np.float).reshape(0,2)

        for i in range(steps):

            img_tmp = torch.FloatTensor([]).view(0, 3, opt.imageSize, opt.imageSize)

            end = (i + 1)*batchSize
            if end > n_sub_imgs:
                end = n_sub_imgs - i*batchSize
            else:
                end = batchSize

            for j in range(end):
                img_tmp = torch.cat((img_tmp, batch[i*batchSize + j]), dim=0)

            if opt.gpu_id >= 0:
                img_tmp = img_tmp.cuda(opt.gpu_id)

            input_v = Variable(img_tmp, requires_grad = False)

            x = vgg_ext(input_v)
            classes, class_ = model(x, random=opt.random)
            output_p = class_.data.cpu().numpy()

            output_pred = np.concatenate((output_pred, output_p), axis=0)

    output_pred = output_pred.mean(0)

    if output_pred[1] >= output_pred[0]:
        pred = 1
    else:
        pred = 0

    return pred, output_pred[1]

transform_fwd = transforms.Compose([
    transforms.Resize((opt.imageSize, opt.imageSize)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def classify_frames(vgg_ext, model, path, label):
    file_lst = get_file_list(path, img_ext_lst)
    file_lst.sort()
    length = len(file_lst)
    frames =[]

    prob_lst = np.array([], dtype=float).reshape(0)
    label_lst = np.array([], dtype=float).reshape(0)

    if (length == 0):
        return
    elif (length == 1):
        print('Error: Only one file!')
        return

    correct = 0
    count_vid = 1
    file_edited = process_file_list(file_lst)
    vid_name = file_edited[0]

    test_img = transform_fwd(Image.open(os.path.join(path, file_lst[0])))
    frames.append(test_img.unsqueeze(0))

    for i in range(1, length):
        if file_edited[i] == vid_name:
            test_img = transform_fwd(Image.open(os.path.join(path, file_lst[i])))
            frames.append(test_img.unsqueeze(0))
        else:
            # clasify the previous frames
            cls, prob = classify_batch(vgg_ext, model, frames)
            if cls == label:
                correct = correct + 1

            label_lst = np.append(label_lst, np.array([label], dtype=float), axis=0)
            prob_lst = np.append(prob_lst, np.array([prob], dtype=float), axis=0)

            # get new items
            del frames
            frames =[]
            vid_name = file_edited[i]
            count_vid = count_vid + 1

            #print(vid_name)
            test_img = transform_fwd(Image.open(os.path.join(path, file_lst[i])))
            frames.append(test_img.unsqueeze(0))


    #classify the last frames
    cls, prob = classify_batch(vgg_ext, model, frames)
    if cls == label:
        correct = correct + 1

    label_lst = np.concatenate((label_lst, np.array([label], dtype=float)), axis=0)
    prob_lst = np.concatenate((prob_lst, np.array([prob], dtype=float)), axis=0)

    print(path)
    print('Number of files: %d' %(length))
    print('Number of videos: %d' %(count_vid))
    print('Number of correct classifications: %d' %(correct))

    return count_vid, correct, label_lst, prob_lst


if __name__ == '__main__':

    path_real = os.path.join(opt.dataset, opt.real)
    path_deepfakes = os.path.join(opt.dataset, opt.deepfakes)
    path_face2face = os.path.join(opt.dataset, opt.face2face)
    path_faceswap = os.path.join(opt.dataset, opt.faceswap)

    vgg_ext = model_big.VggExtractor()
    model = model_big.CapsuleNet(2, opt.gpu_id)

    model.load_state_dict(torch.load(os.path.join(opt.outf,'capsule_' + str(opt.id) + '.pt')))
    model.eval()

    if opt.gpu_id >= 0:
        vgg_ext.cuda(opt.gpu_id)
        model.cuda(opt.gpu_id)

    ###################################################################
    tol_count_vid_real = 0
    tol_correct_real = 0

    tol_count_vid_deepfakes = 0
    tol_correct_deepfakes = 0

    tol_count_vid_face2face = 0
    tol_correct_face2face = 0

    tol_count_vid_faceswap = 0
    tol_correct_faceswap = 0


    # real data
    count_vid, correct, label_lst_global, prob_lst_global = classify_frames(vgg_ext, model, path_real, 0)
    tol_count_vid_real = tol_count_vid_real + count_vid
    tol_correct_real = tol_correct_real + correct

    # deepfakes data
    ount_vid, correct, lb_lst, pb_lst = classify_frames(vgg_ext, model, path_deepfakes, 1)
    tol_count_vid_deepfakes = tol_count_vid_deepfakes + count_vid
    tol_correct_deepfakes = tol_correct_deepfakes + correct

    label_lst_global = np.concatenate((label_lst_global, lb_lst), axis=0)
    prob_lst_global = np.concatenate((prob_lst_global, pb_lst), axis=0)

    # face2face data
    ount_vid, correct, lb_lst, pb_lst = classify_frames(vgg_ext, model, path_face2face, 1)
    tol_count_vid_face2face = tol_count_vid_face2face + count_vid
    tol_correct_face2face = tol_correct_face2face + correct

    label_lst_global = np.concatenate((label_lst_global, lb_lst), axis=0)
    prob_lst_global = np.concatenate((prob_lst_global, pb_lst), axis=0)

    # deepfakes data
    ount_vid, correct, lb_lst, pb_lst = classify_frames(vgg_ext, model, path_faceswap, 1)
    tol_count_vid_faceswap = tol_count_vid_faceswap + count_vid
    tol_correct_faceswap = tol_correct_faceswap + correct

    label_lst_global = np.concatenate((label_lst_global, lb_lst), axis=0)
    prob_lst_global = np.concatenate((prob_lst_global, pb_lst), axis=0)

    tol_count_vid = tol_count_vid_real + tol_count_vid_deepfakes + tol_count_vid_face2face + tol_count_vid_faceswap
    tol_correct = tol_correct_real + tol_correct_deepfakes + tol_correct_face2face + tol_correct_faceswap

    print('##################################')
    print('Number of videos: %d' %(tol_count_vid))
    print('Number of correct classifications: %d' %(tol_correct))
    print('Accuracy: %.2f' %(tol_correct/tol_count_vid*100))

    print('##################################')
    print('Number of correct real classifications: %d' %(tol_correct_real))
    print('Accuracy: %.2f' %(tol_correct_real/tol_count_vid_real*100))

    print('##################################')
    print('Number of correct deepfakes classifications: %d' %(tol_correct_deepfakes))
    print('Accuracy: %.2f' %(tol_correct_deepfakes/tol_count_vid_deepfakes*100))

    print('##################################')
    print('Number of correct face2face classifications: %d' %(tol_correct_face2face))
    print('Accuracy: %.2f' %(tol_correct_face2face/tol_count_vid_face2face*100))

    print('##################################')
    print('Number of correct faceswap classifications: %d' %(tol_correct_faceswap))
    print('Accuracy: %.2f' %(tol_correct_faceswap/tol_count_vid_faceswap*100))

    label_lst_global[label_lst_global > 0] = 1

    fpr, tpr, thresholds = roc_curve(label_lst_global, prob_lst_global, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    print('##################################')
    print('EEr: %.2f' %(eer * 100))
