import time

import matplotlib as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torch.nn.functional as F
from torchvision import transforms, utils
from utils import ToHeatmap, ToTensor

import io
from datetime import datetime


import cv2 as cv

from dplearn.cnn_model import CNNModel
from dplearn.hetmap_loss import HeatmapLoss
from dplearn.ldnet_model import LDNet
from dplearn.unet_model import UNet


from torchvision import transforms


class TrainUnet:
    def __init__(self, data, files, check_pt, cuda_dev, batch_size, num_epochs, htmps_path, best_model_pth, best_ckpt_pth):
        # load data
        #self.lds = landmarks
        #self.images = images
        self.dataset = data
        self.model_path = best_model_pth
        self.best_ckpt_pth = best_ckpt_pth
        self.files = files
        self.check_pt = check_pt
        self.cuda = cuda_dev # 'cuda:0/1'
        self.device = self.init_device(cuda_dev=self.cuda)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        #self.class = 2
        self.htmps_path = htmps_path
        self.optimizer = 'adm'

    
    def init_device(self, cuda_dev='cuda:0'):
        return torch.device(cuda_dev if torch.cuda.is_available() else 'cpu')


    def cuda_info(self, device):
        if device.type == 'cuda':
            dev_idx = int(self.cuda.split(':')[-1])
            print(torch.cuda.get_device_name(dev_idx))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(dev_idx)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(dev_idx)/1024**3,1), 'GB')
        else:
            print("Not a cuda device.")


    def calc_mean_std(self, train_loader):
        train_mean = []
        train_std = []

        for i,image in enumerate(train_loader,0):
            numpy_image = image['image'].numpy()
            batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
            batch_std = np.std(numpy_image, axis=(0, 2, 3))

            train_mean.append(batch_mean)
            train_std.append(batch_std)

        train_mean = torch.tensor(np.mean(train_mean, axis=0))
        train_std = torch.tensor(np.mean(train_std, axis=0))

        print('Mean:', train_mean)
        print('Std Dev:', train_std)
        return train_mean, train_std

    
    def init_transform_ht(self):
        return transforms.Compose([
                    #ToTensor(),
                    ToHeatmap()
                ])


    def data_transform(self, mean, std, train_loader, dataset):
        data_transforms = transforms.Compose([transforms.Normalize(mean=mean, std=std)])
        for i, image in enumerate(train_loader,0):
            #for batch, 
            # print(image)
            
            img = image['image'][0]
            #print(img.shape)
            name = image['landmarks'][0]
            sample = data_transforms(img)
            dataset[i]['image'] = sample
            dataset[i]['landmarks'] = name
        return dataset

    def setup_optimizer(self, model, lr, weight_decay, mom):
        # optim.Adam(model.parameters(), lr = 3e-4, weight_decay = 0.001)

        if self.optimizer == 'sgd':
            print('=========use SGD=========')
            optimizer = optim.SGD([{'params': model.parameters()}], lr=lr, momentum=mom, weight_decay=weight_decay)
        else:
            print('=========use ADAM=========')
            optimizer = optim.Adam([{'params': model.parameters()}], lr=lr, weight_decay=weight_decay)
        return optimizer

    
    def adjust_learning_rate(self, opt, sf):
        print("\nDECAYING learning rate.")
        #if param_group['lr'] < 0.00015:
        #    param_group['lr'] = 0.5
        #else:
        for param_group in opt.param_groups:
            param_group['lr'] = param_group['lr'] * sf
        print("The new learning rate is %f\n" % (opt.param_groups[0]['lr'],))

    
    def save_checkpoint(self, epoch, epochs_since_improvement, model, optimizer, loss, losses, epoch_lst, model_path):
        print('=========== save checkpoint ============')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'loss_list': losses,
            'iter_list': epoch_lst,
        }, model_path)


    
    def calc_loss(self, lds, output_pts, loss_fn, img):
        loss = 0
        #print(img.shape)

        tf = self.init_transform_ht()
        for i,ld in enumerate(lds):
            #print(i)
            #print(ld)
            sample = tf({'image': img[i], 'landmarks': ld})

            for j in range(55):
                ht = sample['landmarks'][j]
                #ht = sample['landmarks']
                #print(ht.shape)
                #print(output_pts.shape)
                #ht = cv.resize(ht, (256, 256))
                ht = torch.from_numpy(ht).to(self.device)               
                ht = ht.type(torch.cuda.FloatTensor)
                loss2 = loss_fn(output_pts[i][j], ht.unsqueeze(0))                    
                loss = loss + loss2
            loss = loss + (loss / self.num_lds)
        loss = loss / img.shape[0]
        #print(lds.shape)
        return loss

    def train_1(self, train_loader, model, loss_fn, optimizer, epoch, best_loss, epoch_since_improv, loss_list, iter_list):
        device = self.device

        losses = np.array([])
        model.train()
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image'].to(device)           
            lds = data['landmarks']
            tf = self.init_transform_ht()

            #print(lds.shape)
            sample = tf({'image': images, 'landmarks': lds[0]})
            _, ht = sample['landmarks']

            images = images.type(torch.cuda.FloatTensor)
            
            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # forward pass to get outputs
            output_pts = model(images)

            print(output_pts.shape)
            print(ht.shape)
            print(torch.from_numpy(ht).unsqueeze(0).shape)
            #print(ht)

            loss = loss_fn(output_pts, torch.from_numpy(ht).unsqueeze(0).long().to(device))

            step_loss = loss.item()

            # backward pass to calculate the weight gradients
            loss.backward()
            # update the weights
            optimizer.step()

            losses = np.append(losses, step_loss)

            if batch_i % 30 == 0:    # print every 10 batches
                #loss_list.append(loss.data)
                #iter_list.append(cnt)
                print("==============BATCH================")
                print('Epoch: {}, Batch: {}, Loss: {}, Avg. Loss: {}\n'.format(epoch + 1, batch_i+1, loss.data, (sum(list(losses))/len(list(losses)))))

                #Additional Info when using cuda
                self.cuda_info(device)

            #l = np.mean(losses)
            if step_loss < best_loss:
                print('============= loss down =============')
                best_loss = step_loss
                epoch_since_improv = 0
                self.save_checkpoint(epoch, epoch_since_improv, model, optimizer, best_loss, loss_list, iter_list, self.best_ckpt_pth)
            #else:
            #    #print('============== loss not improvement ============ ')
            #    #epoch_since_improv += 1 
            #    print('============== loss not improvement ============ ')
            #    epoch_since_improv += 1 
            #cnt += 1
            torch.cuda.empty_cache()
        
        return np.mean(losses), epoch_since_improv, step_loss


    def train_model(self):
        # setting device on GPU if available, else CPU
        device = self.device
        print('Using device:', device)
        print()

        loss_list = []
        iter_list = []
        loss = 0
        epoch_s = 0
        best_loss = float('inf')

        #Additional Info when using cuda
        self.cuda_info(device)

        # organize the dataset: {'image':...,'landmarks':...}
        dataset = self.dataset        

        # load data in batches
        batch_size = self.batch_size
        train_loader = DataLoader(dataset, 
                                batch_size=batch_size,
                                shuffle=True, 
                                num_workers=0)

        # calc mean and std & transform dataset
        train_mean, train_std = self.calc_mean_std(train_loader)
        dataset = self.data_transform(train_mean, train_std, train_loader, dataset)

        # init CNN
        model = UNet() # LDNet()
        model.to(device)

        """ LOSS FNS:
        https://medium.com/udacity-pytorch-challengers/a-brief-overview-of-loss-functions-in-pytorch-c0ddb78068f7
        """
        #loss_fn = nn.MSELoss().to(device)
        # OR: l2 norm loss
        loss_fn = nn.CrossEntropyLoss()
        #HeatmapLoss().to(device)
        #loss_fn = nn.L1Loss(reduction='mean').to(device)
        
        # adam optimizer
        optimizer = optim.RMSprop(model.parameters(), lr=0.0001, weight_decay=1e-8, momentum=0.9)
        # self.setup_optimizer(model, 3e-4, 0.001, 0)
        # The l2 regularization simply penalizes large weights and improves the network’s performance. == weight_decay
        
        num_epochs = self.num_epochs   
        # load checkpoint     

        if self.check_pt:
            print("=====load chkpt======")
            checkpoint = torch.load(self.model_path, map_location=self.cuda)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_s = checkpoint['epoch']
            start_epoch = epoch_s
            num_epochs = epoch_s + 500
            best_loss = checkpoint['loss']

        ############################# train model ##############################

        # turn on train mode
        model.train()
        start_time = time.time()
        cnt = 0        
       
        s = 0
        epoch_since_improv = 0

        for epoch in range(epoch_s, num_epochs, 1):
            #if epoch_since_improv == 100:
            #    break
            #if epoch_since_improv != 0 and epoch_since_improv % 10 == 0:
            #    print('============= reload model, adjust lr ===============')
            #    checkpoint = torch.load(self.best_ckpt_pth)
            #    model.load_state_dict(checkpoint['model_state_dict'])
            #    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #    best_loss = checkpoint['loss']
            #    self.adjust_learning_rate(optimizer, 0.5)

            loss, epoch_since_improv, step_loss = self.train_1(train_loader, model, loss_fn, optimizer, epoch, best_loss, epoch_since_improv, loss_list, iter_list)
            loss_list.append(loss)
            iter_list.append(epoch)
            print('==== avg lose of epoch {0} is {1} ====='.format(epoch, loss))
            if loss < best_loss:
                print('============= loss down =============')
                best_loss = loss
                epoch_since_improv = 0
                self.save_checkpoint(epoch, epoch_since_improv, model, optimizer, best_loss, loss_list, iter_list, self.best_ckpt_pth)
            else:
                print('============== loss not improvement ============ ')
                epoch_since_improv += 1

    
   


""" torch.save({
        #'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'best_pred': output_pts,
        'loss_list': loss_list,
        'iter_list': iter_list,
    }, "models/model_earrgbl2end.pth")
"""
