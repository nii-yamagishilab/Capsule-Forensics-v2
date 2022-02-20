"""
Copyright (c) 2018, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------
Script for data preprocessing
"""

import argparse
import cv2
import numpy as np
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('--root', default ='path_to_FaceForensicsPP')
parser.add_argument('--mask_ref', default ='FaceSwap')
parser.add_argument('--type', default ='NeuralTextures')
parser.add_argument('--list', default ='path_to_FaceForensicsPP/splits/train.json')
parser.add_argument('--output', default ='path_to_output/4_neuraltextures')
parser.add_argument('--limit', type=int, default=100, help='number of images to extract for each video')
parser.add_argument('--scale', type=float, default=1.3, help='enables resizing')

opt = parser.parse_args()
print(opt)

def to_bw(mask, thresh_binary=10, thresh_otsu=255):
    im_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(im_gray, thresh_binary, thresh_otsu, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return im_bw

def get_bbox(mask, thresh_binary=10, thresh_otsu=255):
    im_bw = to_bw(mask, thresh_binary, thresh_otsu)

    # im2, contours, hierarchy = cv2.findContours(im_bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(im_bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    locations = np.array([], dtype=np.int).reshape(0, 5)

    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
        else:
            cX = 0
        if M["m00"] > 0:
            cY = int(M["m01"] / M["m00"])
        else:
            cY = 0

        # calculate the rectangle bounding box
        x,y,w,h = cv2.boundingRect(c)
        locations = np.concatenate((locations, np.array([[cX, cY, w, h, w + h]])), axis=0)

    if locations.size == 0:
        return None

    max_idex = locations[:,4].argmax()
    bbox = locations[max_idex, 0:4].reshape(4)
    return bbox

def extract_face(image, bbox, scale = 1.0):
    h, w, d = image.shape
    radius = int(bbox[3] * scale / 2)

    y_1 = bbox[1] - radius
    y_2 = bbox[1] + radius
    x_1 = bbox[0] - radius
    x_2 = bbox[0] + radius

    if x_1 < 0:
        x_1 = 0
    if y_1 < 0:
        y_1 = 0
    if x_2 > w:
        x_2 = w
    if y_2 > h:
        y_2 = h

    crop_img = image[y_1:y_2, x_1:x_2]

    return crop_img

def extract_face_videos(filename, compress):
    print(compress + '_' + filename)

    input_vid_path = os.path.join(opt.root, 'manipulated_sequences', opt.type, compress, 'videos')
    input_mask_ref = os.path.join(opt.root, 'manipulated_sequences', opt.mask_ref, 'masks', 'videos')

    vidcap_vid = cv2.VideoCapture(os.path.join(input_vid_path, filename + '.mp4'))
    success_cap_vid, image_cap = vidcap_vid.read()

    vidcap_mask_ref = cv2.VideoCapture(os.path.join(input_mask_ref, filename + '.mp4'))
    success_cap_mask_ref, image_mask_ref = vidcap_mask_ref.read()

    count = 0
    skip = 0

    while (success_cap_vid and success_cap_mask_ref):

        bbox = get_bbox(image_mask_ref)

        if bbox is None:
            count += 1
            skip += 1

            success_cap_vid, image_cap = vidcap_vid.read()
            success_cap_mask_ref, image_mask_ref = vidcap_mask_ref.read()

            continue

        image_cropped = extract_face(image_cap, bbox, opt.scale)

        if image_cropped is not None:
            cv2.imwrite(os.path.join(opt.output, compress + '_' + filename + "_%d_neuraltextures.jpg" % count), image_cropped)
            count += 1

        if count - skip >= opt.limit:
            break

        success_cap_vid, image_cap = vidcap_vid.read()
        success_cap_mask_ref, image_mask_ref = vidcap_mask_ref.read()

if __name__ == '__main__':

    compress = ['raw', 'c23', 'c40']

    json_file = open(opt.list, 'r')
    json_txt = json_file.read()
    vid_list = json.loads(json_txt)

    for i in range(len(compress)):
        for j in range(len(vid_list)):
            filename = vid_list[j][0] + '_' + vid_list[j][1]
            extract_face_videos(filename, compress[i])

            filename = vid_list[j][1] + '_' + vid_list[j][0]
            extract_face_videos(filename, compress[i])
