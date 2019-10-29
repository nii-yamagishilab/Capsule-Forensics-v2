"""
Copyright (c) 2019, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------
Script for testing Capsule-Forensics-v2 on CGvsPhoto database
"""

import sys
sys.setrecursionlimit(15000)
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torch.autograd import Variable
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import model_big

import math
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default ='databases/cgvsphoto_full', help='path to dataset')
parser.add_argument('--test_set', default ='test', help='path to test dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--imageSize', type=int, default=100, help='the height / width of the input image to network')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--outf', default='checkpoints/cgvsphoto', help='folder to output images and model checkpoints')
parser.add_argument('--random_sample', type=int, default=0, help='number of random sample to test')
parser.add_argument('--random', action='store_true', default=False, help='enable randomness for routing matrix')
parser.add_argument('--id', type=int, default=41, help='checkpoint ID')

opt = parser.parse_args()
print(opt)


if __name__ == '__main__':

    text_writer = open(os.path.join(opt.outf, 'test_full.txt'), 'w')

    transform_fwd = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    # folder dataset
    dataset_test = dset.ImageFolder(root=os.path.join(opt.dataset, opt.test_set), transform=transform_fwd)
    assert dataset_test
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=int(opt.workers))


    def extract_subimages(image, subimage_size=100):
        #print(image.shape)
        subimages = []
        width = image.shape[3]
        height = image.shape[2]

        current_height = 0

        while current_height + subimage_size <= height:
            current_width = 0
            while current_width + subimage_size <= width:
                sub = image[:,:,current_height:current_height+subimage_size, current_width:current_width+subimage_size]

                subimages.append(sub)
                current_width += subimage_size
            current_height += subimage_size

        return subimages

    vgg_ext = model_big.VggExtractor()
    capnet = model_big.CapsuleNet(2, opt.gpu_id)

    capnet.load_state_dict(torch.load(os.path.join(opt.outf,'capsule_' + str(opt.id) + '.pt')))
    capnet.eval()

    if opt.gpu_id >= 0:
        vgg_ext.cuda(opt.gpu_id)
        capnet.cuda(opt.gpu_id)


    ##################################################################################
    predict_lst = np.array([], dtype=np.float)
    #prob_lst = np.array([], dtype=np.float)
    labels_lst = np.array([], dtype=np.float)

    for img_data, labels_data in dataloader_test:

        img_label = labels_data.numpy().astype(np.float)

        subimages = extract_subimages(img_data, opt.imageSize)
        prob = np.array([[0.0, 0.0]])
        n_sub_imgs = len(subimages)

        if (opt.random_sample > 0):
            if n_sub_imgs > opt.random_sample:
                np.random.shuffle(subimages)
                n_sub_imgs = opt.random_sample

            img_tmp = torch.FloatTensor([]).view(0, 3, opt.imageSize, opt.imageSize)

            for i in range(n_sub_imgs):
                img_tmp = torch.cat((img_tmp, subimages[i]), dim=0)

            if opt.gpu_id >= 0:
                img_tmp = img_tmp.cuda(opt.gpu_id)

            input_v = Variable(img_tmp, requires_grad = False)

            x = vgg_ext(input_v)
            classes, class_ = capnet(x, random=opt.random)
            output_pred = class_.data.cpu().numpy()

        else:
            batchSize = 10
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
                    img_tmp = torch.cat((img_tmp, subimages[i*batchSize + j]), dim=0)

                if opt.gpu_id >= 0:
                    img_tmp = img_tmp.cuda(opt.gpu_id)

                input_v = Variable(img_tmp, requires_grad = False)

                x = vgg_ext(input_v)
                classes, class_ = capnet(x, random=opt.random)
                output_p = class_.data.cpu().numpy()

                output_pred = np.concatenate((output_pred, output_p), axis=0)

        output_pred = output_pred.mean(0)

        if output_pred[1] >= output_pred[0]:
            pred = 1.0
        else:
            pred = 0.0

        #print('%d - %d' %(pred, img_label))
        text_writer.write('%d,%.2f\n' % (img_label, output_pred[1]))

        predict_lst = np.concatenate((predict_lst, np.array([pred])), axis=0)
        #prob_lst = np.concatenate((prob_lst, output_pred[1]), axis=0)
        labels_lst = np.concatenate((labels_lst, img_label), axis=0)


    acc = metrics.accuracy_score(labels_lst, predict_lst)

    print(len(predict_lst))
    print('%d\t%.4f' % (opt.id, acc))


    text_writer.flush()
    text_writer.close()
