from read_data import ReadData
from write_data import WriteData
from sop import SOP
from utils import Visualize, Normalize, Crop, Rescale

import argparse

# from sklearn.neighbors import NearestNeighbors
from torchvision import transforms, utils

from dplearn.train_model import TrainModel
from dplearn.test_model import TestModel
#from dplearn.train_unet import TrainUnet
#from dplearn.test_unet import TestUnet


import matplotlib as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torch.nn.functional as F

#import tensorflow as tf
#import tensorflow_datasets as tfds


def parse_args():
    parser = argparse.ArgumentParser(description='Train/test ear network')
    # general
    parser.add_argument('--train_path', type=str, default='db-with-landmark/CollectionA/augmented_data/train/', help='checkpoint')
    parser.add_argument('--end_epoch', type=int, default=1000, help='training epoch size.')
    parser.add_argument('--lr', type=float, default=4e-6, help='start learning rate')
    parser.add_argument('--optimizer', default='adm', help='optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in each context')
    parser.add_argument('--checkpoint', type=str, default='BEST_checkpoint.tar', help='checkpoint')
    parser.add_argument('--print_freq', type=int, default=100, help='checkpoint')
    parser.add_argument('--shrink_factor', type=float, default=0.5, help='checkpoint')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    
    # READ DATA PATHS
    #train_path_2D = 'db-with-landmark/CollectionA/train/'
    #test_path_2D = 'db-with-landmark/CollectionA/test/'
    train_path_2D = 'db-with-landmark/CollectionA/augmented_data/train/o_images/'
    htmps_ears_train = 'db-with-landmark/CollectionA/augmented_data/train/o_landmarks/'

    test_path_2D = 'db-with-landmark/CollectionA/augmented_data/test/o_images/'
    model_path_3DMM = 'yorkEarModel/Model/earmodel.mat'
    ptc_path = 'yorkEarModel/Data/ply/'
    test_faces_ld = '300W/Test/300W_test.txt'
    test_faces = '300W/Test/'
    train_faces = '300W/Train/'
    train_faces_ld = '300W/Train/300W_train.txt'
    #htmps_ears_train = 'db-with-landmark/CollectionA/htmps_train/'
    #htmps_ears_test = test_path_2D + 'heatmaps/'    
    htmps_ears_test = 'db-with-landmark/CollectionA/augmented_data/test/o_landmarks/'
    htmps_face_train = train_faces + 'heatmaps/'
    htmps_face_test = test_faces + 'heatmaps/'

    # initialize read data class
    rd = ReadData(train_path_2D, ptc_path, model_path_3DMM, htmps_ears_train, reduced=False, transform=True, augment=False, model='hourglass')

    # WRITE DATA PATHS
    save_ptc_path = 'generated_data/ptc/'
    save_imgs_path = 'generated_data/images/'
    save_2D_lds_path = 'generated_data/landmarks_2D/'
    save_3D_lds_path = 'generated_data/landmarks_3D/'

    # initialize write data class
    #sv = WriteData(save_ptc_path, save_imgs_path, save_2D_lds_path, save_3D_lds_path)

    # load images and landmarks
    #rd.read_images()

    #print(rd.htmps_path)
    #rd.htmps_path = htmps_face_train
    #print(rd.htmps_path)
    #rd.read_300w(train_faces, train_faces_ld)
    #rd.htmps_path = htmps_face_test
    #print(rd.htmps_path)
    #rd.read_300w(test_faces, test_faces_ld



    ## transform image dataset
    rd_test = ReadData(test_path_2D, ptc_path, model_path_3DMM, htmps_ears_test,augment=False, transform=True, t_t='test')
    rd_test.read_awe()


    #tm = TrainModel(rd.dataset, rd.files, True, 'cuda:1', 20, 500, htmps_ears_train, 'models/model_2stack.pth', 'models/model_2stack.pth')
    #rd_test = ReadData(test_path_2D, ptc_path, model_path_3DMM, htmps_ears_test, transform=True, augment=True, t_t='test')
    #rd_test.read_images()


    #tm = TrainModel(rd.dataset, rd.files, False, 'cuda:0', 1, 500, htmps_ears_train, 'models/unet-ld.pth', 'models/unet-ld.pth')
    #tm = TrainModel(rd.dataset, rd.files, False, 'cuda:1', 15, 500, htmps_ears_train, 'models/model_2stack.pth', 'models/hourglass_2stack.pth')

    # data, files, check_pt, cuda_dev, batch_size, num_epochs, htmps_path):
    #tm.train_model()

    #vs = Visualize()
    #unet_train = TrainUnet(rd.dataset, rd.files, False, 'cuda:0', 1, 500, htmps_ears_train, 'models/unet_12k.pth', 'models/unet_12k.pth')
    #unet_train.train_model()
    print("==TEST DATA==")

    tm2 = TestModel(rd_test.landmarks, rd_test.images, rd_test.dataset, test=True)
    tm2.predict_keypoints()

    #unet_test = TestUnet(rd_test.landmarks, rd_test.images, rd_test.dataset, test=True)
    #unet_test.predict_keypoints()

    print("==VALIDATION DATA==")

    #tm3 = TestModel(rd.landmarks, rd.images, rd.dataset, test=False)
    #tm3.predict_keypoints()

    # printing out the dimensions of the data to see if they make sense
    #print(test_images.data.size())
    #print(test_outputs.data.size())
    #print(gt_pts.size())
    #print(test_outputs)
    #print(gt_p)
    #vs.visualize_output(test_images, test_outputs, gt_pts)


    # read PLY
    # rd.read_PTC()

    # load 3DMM
    # rd.read_3DMM()

    # initilaize SOP class and fit landmarks for each pointcloud
    """"for idx,name in enumerate(rd.ptc):
        print(idx, name)
        landmarks = rd.landmarks[idx]
        morphable = rd.ptc[name]
        landmarks_3D = rd.landmarks_3D[name]
        sop = SOP(landmarks, morphable, landmarks_3D)

        # perform FITTING
        min_res, min_coor, min_idxs = sop.ransac(name)

        # augment data
        sop.augment_data(min_res, min_coor, min_idxs, name)


    """

    # TODO:
    # transform data for CNN
    #rd = ReadData(train_path_2D, ptc_path, model_path_3DMM)


    ## transform dataset


    """vs = Visualize()

    print(rd.images)
    print(rd.landmarks)

    vs.show_landmarks(rd.images['train_0000'], rd.landmarks[0])

    cmp_tf = transforms.Compose({
        Normalize()
    })

    composed_transform = transforms.Compose([
        Rescale((224, 224))
    ])

    composed_transform2 = transforms.Compose([
        Crop(rd.ears_bbox['train_0000'])
    ])

    img = rd.images['train_0000']
    lds = rd.landmarks[0]

    transformed = cmp_tf({'image': rd.images['train_0000'], 'landmarks': rd.landmarks[0]})

    vs.show_landmarks(transformed['image'], transformed['landmarks'])"""


