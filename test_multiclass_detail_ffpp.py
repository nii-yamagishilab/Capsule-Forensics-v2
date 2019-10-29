"""
Copyright (c) 2019, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------
Script for testing Capsule-Forensics-v2 on FaceForensics++ database with detail results on each class (Real, DeepFakes, Face2Face, FaceSwap)
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
import model_big

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default ='databases/faceforensicspp', help='path to dataset')
parser.add_argument('--test_set', default ='test', help='test set')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imageSize', type=int, default=300, help='the height / width of the input image to network')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--outf', default='checkpoints/multiclass_faceforensicspp', help='folder to output model checkpoints')
parser.add_argument('--random', action='store_true', default=False, help='enable randomness for routing matrix')
parser.add_argument('--id', type=int, default=13, help='checkpoint ID')

opt = parser.parse_args()
print(opt)

if __name__ == '__main__':

    text_writer = open(os.path.join(opt.outf, 'test_detail.txt'), 'w')

    transform_fwd = transforms.Compose([
        transforms.Resize((opt.imageSize, opt.imageSize)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    dataset_test = dset.ImageFolder(root=os.path.join(opt.dataset, opt.test_set), transform=transform_fwd)
    assert dataset_test
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))

    vgg_ext = model_big.VggExtractor()
    capnet = model_big.CapsuleNet(4, opt.gpu_id)

    capnet.load_state_dict(torch.load(os.path.join(opt.outf,'capsule_' + str(opt.id) + '.pt')))
    capnet.eval()

    if opt.gpu_id >= 0:
        vgg_ext.cuda(opt.gpu_id)
        capnet.cuda(opt.gpu_id)


    ##################################################################################

    tol_label = np.array([], dtype=np.float)
    tol_pred = np.array([], dtype=np.float)

    for img_data, labels_data in tqdm(dataloader_test):

        img_label = labels_data.numpy().astype(np.float)

        if opt.gpu_id >= 0:
            img_data = img_data.cuda(opt.gpu_id)
            labels_data = labels_data.cuda(opt.gpu_id)

        input_v = Variable(img_data)

        x = vgg_ext(input_v)
        classes, class_ = capnet(x, random=opt.random)

        output_dis = class_.data.cpu()
        output_pred = torch.argmax(output_dis, dim=1).numpy()

        tol_label = np.concatenate((tol_label, img_label), axis=0)
        tol_pred = np.concatenate((tol_pred, output_pred), axis=0)


    real_total = 0
    real_correct = 0
    deepfakes_total = 0
    deepfakes_correct = 0
    face2face_total = 0
    face2face_correct = 0
    faceswap_total = 0
    faceswap_correct = 0

    for i in range(len(tol_label)):
        if tol_label[i] == 0:
            real_total += 1
            if tol_pred[i] == tol_label[i]:
                real_correct += 1
        elif tol_label[i] == 1:
            deepfakes_total += 1
            if tol_pred[i] == tol_label[i]:
                deepfakes_correct += 1
        elif tol_label[i] == 2:
            face2face_total += 1
            if tol_pred[i] == tol_label[i]:
                face2face_correct += 1
        elif tol_label[i] == 3:
            faceswap_total += 1
            if tol_pred[i] == tol_label[i]:
                faceswap_correct += 1

    real_acc = real_correct / real_total * 100
    deepfakes_acc = deepfakes_correct / deepfakes_total * 100
    face2face_acc = face2face_correct / face2face_total * 100
    faceswap_acc = faceswap_correct / faceswap_total * 100

    print('[Epoch %d] - Real: %.2f, Deepfakes: %.2f, Face2face: %.2d, FaceSwap: %.2f' % (opt.id, real_acc, deepfakes_acc, face2face_acc, faceswap_acc))
    text_writer.write('%d,%.2f,%.2f,%.2f,%.2f\n'% (opt.id, real_acc, deepfakes_acc, face2face_acc, faceswap_acc))

    text_writer.flush()
    text_writer.close()
