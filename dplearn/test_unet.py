import io

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils import ToHeatmap, Visualize

from dplearn.cnn_model import CNNModel
from dplearn.hetmap_loss import HeatmapLoss
from dplearn.ldnet_model import LDNet
from dplearn.unet_model import UNet
from utils import ToTensor

from sklearn import preprocessing

tb = SummaryWriter(comment="test")


class TestModel:
    # model_aug12K_best_l2_fixedratio
    def __init__(self, landmarks, images, data, test, path='./models/model_3stack.pth'):
        # load data
        self.lds = landmarks
        self.images = images
        self.dataset = data
        self.PATH = path
        self.num_lds = 55
        self.net = LDNet()
        self.net_dict = torch.load(self.PATH,map_location='cuda:0')
        self.loss_lst = self.net_dict['loss_list']
        self.iter_lst = self.net_dict['iter_list']
        self.epoch = self.net_dict['epoch']
        self.net.load_state_dict(self.net_dict['model_state_dict'],strict=False)
        self.net.eval()
        self.vs = Visualize()
        self.test = test
        print(len(self.dataset))


    def root_mean_squared_error(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_pred - y_true)))

    
    def init_transform_ht(self):
        return transforms.Compose([
                    #ToTensor(),
                    ToHeatmap()
                ])

    
    def predict_keypoints(self):
        # organize the dataset: {'image':...,'landmarks':...}
        dataset = self.dataset
        batch_size = 1
        vs = self.vs

        test_loader = DataLoader(dataset, 
                                batch_size=1,
                                shuffle=False, 
                                num_workers=0)

        #sprint(self.epoch)

        """test_mean = []
        test_std = []

        for i,image in enumerate(test_loader,0):
            numpy_image = image['image'].numpy()
            batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
            batch_std = np.std(numpy_image, axis=(0, 2, 3))

            test_mean.append(batch_mean)
            test_std.append(batch_std)

        test_mean = torch.tensor(np.mean(test_mean, axis=0))
        test_std = torch.tensor(np.mean(test_std, axis=0))

        print('Mean:', test_mean)
        print('Std Dev:', test_std)

        data_transforms = transforms.Compose([transforms.Normalize(mean=test_mean, std=test_std)])
        for i,image in enumerate(test_loader,0):
            img = image['image'][0]
            name = image['landmarks'][0]

            #print(image)
            #print(name)
            #print(img)
            sample = data_transforms(img)
            #print(sample.shape)
            dataset[i]['image'] = sample
            dataset[i]['landmarks'] = name

            #print(dataset[i])
            #print(sample)"""

        images = np.array([])
        lds2 = np.empty((0,2), int)
        gt_lds_all = np.empty((0,2), int)

        samples = []
        mse = []
        l2 = []
        rmse = []

        ct = 0

        #loss_fn = nn.MSELoss()

        cum_lss = [0]
        cum_lss_ld = []

        scores = []

        # iterate through the test dataset
        for i, sample in enumerate(test_loader):
            # get sample data: images and ground truth landmarks
            img = sample['image']
            print(img.shape)
            gt_lds = sample['landmarks']
            #gt_lds = gt_lds.data.numpy()[0]

            #lds = np.zeros((55, sample['image'].shape[2], sample['image'].shape[3]))

            # convert images to FloatTensors
            image_tensor = img.type(torch.FloatTensor)


            # forward pass to get net output
            output_pts = self.net(image_tensor).permute(0, 2, 3, 1).squeeze().detach().numpy()
            #print(len(output_pts))
            #print("prediction")

            fg = output_pts.argmax(2) * 255

            tf = self.init_transform_ht()
            sample = tf({'image': img[0], 'landmarks': gt_lds[0]})
            _, label = sample['landmarks']

            output_array_ = (fg - np.min(fg)) / (np.max(fg) - np.min(fg))
            label_array = (label - np.min(label)) / (np.max(label) - np.min(label))        

            intersection = np.logical_and(label_array, output_array_)
            union = np.logical_or(label_array, output_array_)
            iou_score = np.sum(intersection) / np.sum(union)
            print('IOU: ', iou_score)
            scores.append(iou_score)

            cv.imwrite('results/'+str(ct)+'_label.png', label.astype(np.uint8))
            cv.imwrite('results/'+str(ct)+'.png', fg.astype(np.uint8))

            ct += 1

        # visualization of training loss
        plt.plot(scores, ct)
        plt.xlabel("Sample")
        plt.ylabel("IOU score")
        plt.show()
        plt.pause(0.001)
        plt.savefig('iou_unet_score.png')
        plt.close()

        """

        #mse = mean_squared_error(gt_lds_all, lds2)
        #r_square = r2_score(gt_lds_all, lds2)
        # axis=1  MSE za landmarke za posamezno sliko
        print('MSE - by test sample: ', ((gt_lds_all - lds2)**2).mean(axis=1).mean())
        #print(np.reshape((gt_lds_all - lds2)**2, (30,55,2)))
        #print('MSE - by ld: ', ((gt_lds_all - lds2)**2))

        print('RMSE: ', self.root_mean_squared_error(gt_lds_all, lds2))

        # axis=0  MSE za specific landmark
        print('MSE - by landmark: ' , ((gt_lds_all - lds2)**2).mean(axis=0).mean())

        print("R2 score: ", r2_score(gt_lds_all, lds2) * 100)
        """
