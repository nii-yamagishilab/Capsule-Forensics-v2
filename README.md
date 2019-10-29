# Capsule-Forensics

Implementation of the paper:  <a href="https://arxiv.org/abs/1810.11215">Capsule-Forensics: Using Capsule Networks to Detect Forged Images and Videos</a> (ICASSP 2019).

You can clone this repository into your favorite directory:

    $ git clone https://github.com/nii-yamagishilab/Capsule-Forensics

## Requirement
- PyTorch 0.3
- TorchVision
- scikit-learn
- Numpy
- pickle

## Project organization
- Datasets folder, where you can place your training, evaluation, and test set:

      ./dataset/<train; validation; test>
- Checkpoint folder, where the training outputs will be stored:

      ./checkpoints

Pre-trained models for Face2Face and DeepFakes detection (with settings described in our paper) are provided in the checkpoints folder.

## Dataset
Each dataset has two parts:
- Real images: ./dataset/\<train;test;validation\>/Real
- Fake images: ./dataset/\<train;test;validation\>/Fake

The dataset need to be pre-processed to crop facial area.

## Training
**Note**: Parameters with detail explanation could be found in the corresponding source code.

    $ python train.py --dataset dataset --train_set train --val_set validation --outf checkpoints --batchSize 100 --niter 100

## Evaluating
**Note**: Parameters with detail explanation could be found in the corresponding source code.

For testing on image level, using test.py

    $ python test.py --dataset dataset --test_set test --outf checkpoints --id <your selected id>
    
For testing on large images using patch aggregation strategy, please use `test_by_patches.py`.

For testing on video level by aggregating the predicted probabilities of video frames, please use `test_vid_lvl.py`

## Authors
- Huy H. Nguyen (https://researchmap.jp/nhhuy/?lang=english)
- Junichi Yamagishi (https://researchmap.jp/read0205283/?lang=english)
- Isao Echizen (https://researchmap.jp/echizenisao/?lang=english)

## Acknowledgement
This work was supported by JSPS KAKENHI Grant Numbers (16H06302, 17H04687, 18H04120, 18H04112, 18KT0051) and by JST CREST Grant Number JPMJCR18A6, Japan.

## Reference
H. H. Nguyen, J. Yamagishi, and I. Echizen, “Capsule-Forensics: Using Capsule Networks to Detect Forged Images and Videos,” Proc. of the 2019 International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2019), 5 pages, (May 2019)
