import cv2 as cv
import matplotlib.pyplot as plt
import time
import numpy as np
from numpy.lib.function_base import angle
from numpy.ma.core import minimum_fill_value
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage.interpolation import rotate
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchvision import transforms, utils
from utils import (Augment, Crop, Equalize, Normalize, Rescale, ToHeatmap,
                   ToTensor, Visualize)

from dplearn.cnn_model import CNNModel
from dplearn.hetmap_loss import HeatmapLoss
from dplearn.ldnet_model import LDNet

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import os

import sys

writer = SummaryWriter('data/model_logs/')





class TestModel:
    def __init__(self, landmarks, images, data, test, path='./models/hourglass_2stack.pth'):
        # load data
        self.lds = landmarks
        self.images = images
        self.dataset = data
        self.PATH = path
        self.net = LDNet()
        #print(torch.load(self.PATH))
        self.net.load_state_dict(torch.load(self.PATH, map_location='cuda:0')['model_state_dict'], strict=False)
        
        #print(self.net)

        """pytorch_total_params = sum(p.numel() for p in self.net.parameters())
        pytorch_tl_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)

        print(pytorch_total_params)
        print(pytorch_tl_params)"""
        self.net.eval()

        self.vs = Visualize()
        self.test = test
        self.train_loss = (torch.load(self.PATH, map_location='cuda:0'))['loss_list']
        self.iter_list = (torch.load(self.PATH, map_location='cuda:0'))['iter_list']
        self.standard_lds = np.asarray([[175,54],[164,24],[141,5],[116,4],[100,16],[84,28],[73,44],[68,68],[64,96],[72,128],
                                        [84,152],[88,164],[96,176],[104,188],[120,200],[128,208],[136,219],[144,228],[164,236],[172,225],
                                        [147,101],[153,90],[153,61],[152,40],[136,32],[120,28],[104,28],[96,36],[84,68],[82,90],[81, 104],
                                        [83,113],[92,136],[100,148],[112,160],[168,111],[168,136],[168,148],[176,160],[176,172],[168,164],
                                        [152,144],[136,144],[124,140],[116,132],[107,110],[118,94],[128,88],[148,80],[156,80],[96,84],
                                        [100,72], [101,64], [108,  48], [116,36]])


        img = np.ones((256,256,3))
        img = img * 255
        """plt.rcParams["font.family"] = "Times New Roman"
        #plt.figure(frameon=False).patch.set_facecolor('black')
        plt.imshow((img).astype(np.uint8))
        #plt.patch
        plt.scatter(self.standard_lds[:, 0], self.standard_lds[:, 1], s=20, marker='.', c='black')
        for i in range(len(self.standard_lds)):
            x = self.standard_lds[i][0]
            y = self.standard_lds[i][1] + 5
            plt.text(x+2, y, str(i), color="cornflowerblue", fontsize=12)
        plt.axis('off')
        #plt.savefig('std_ear_svg.svg',bbox_inches='tight', pad_inches=0, transparent=True,format="svg")
        #plt.show()"""

        avg_lds = np.asarray([[165.23409669211196, 52.469465648854964],
[155.23918575063612, 44.61132315521628],
[142.543893129771, 36.01017811704835],
[125.36577608142494, 31.708015267175572],
[108.63931297709924, 37.34796437659033],
[97.30470737913485, 49.13104325699746],
[90.04389312977099, 65.60496183206106],
[87.31552162849873, 86.0998727735369],
[87.58651399491094, 100.96628498727735],
[90.46374045801527, 121.56170483460559],
[93.31615776081425, 137.1533078880407],
[97.53944020356234, 153.27162849872772],
[103.25508905852418, 168.1405852417303],
[110.10050890585242, 180.97519083969465],
[119.19338422391857, 195.27989821882952],
[127.96119592875318, 206.96882951653944],
[140.06043256997455, 216.8148854961832],
[155.71882951653944, 221.66730279898218],
[166.941475826972, 220.9586513994911],
[177.5356234096692, 214.22455470737913],
[166.6698473282443, 114.12468193384224],
[164.51208651399492, 98.07760814249365],
[160.64567430025446, 79.15458015267176],
[151.75763358778627, 63.37468193384224],
[139.3969465648855 ,52.39312977099237],
[125.20356234096693, 47.92684478371501],
[111.56424936386769, 52.4529262086514],
[104.25763358778626, 62.55152671755725],
[98.67175572519083, 82.91412213740458],
[98.93893129770993, 101.80788804071247],
[100.4147582697201, 116.02162849872774],
[101.88358778625954, 131.11259541984734],
[105.19083969465649, 144.51844783715012],
[110.23027989821882 ,154.63104325699746],
[118.11895674300254, 163.9058524173028],
[174.6946564885496, 123.20165394402035],
[169.68066157760813, 133.23346055979644],
[166.6615776081425, 148.03180661577608],
[172.1412213740458, 159.13295165394402],
[173.0966921119593, 169.9325699745547],
[161.31679389312978, 165.40076335877862],
[152.42748091603053, 155.41094147582697],
[138.83269720101782, 152.46882951653944],
[127.95801526717557, 146.19656488549617],
[123.0114503816794 ,135.34923664122138],
[124.81552162849873, 122.76717557251908],
[130.9917302798982 ,112.9910941475827],
[139.8301526717557 ,106.69529262086515],
[149.45483460559797, 102.40076335877863],
[160.68575063613233, 98.93765903307889],
[104.37786259541984 ,110.36832061068702],
[108.28880407124682 ,100.4885496183206],
[114.01208651399492 ,84.9204834605598],
[119.63104325699746 ,71.78307888040712],
[127.06234096692111, 59.30279898218829]])

        print(avg_lds)
        print(avg_lds.shape)
        print(avg_lds[:,0])
        print(avg_lds[:,1])
        img = np.ones((256,256,3))
        img = img * 255

        #plt.figure()

        #plt.figure(frameon=False).patch.set_facecolor('black')
        #plt.imshow((img).astype(np.uint8))
        #plt.scatter(avg_lds[:,0], avg_lds[:,1])
        #plt.show()
 

        
        #np.asarray([[162,72],[158,66],[154,51],[140,35],[129,27],[116,31],[103,45],[98,63],[94,82],[92,102],
        #                    [89,123],[90, 144],[93,160],[99,177],[104,196],[113,208],[121,217],[132,223],[140,227],
        #                    [152,230],[162,231],[155,131],[154,109],[151,87],[146,74],[134,64],[124,58],[110,57],[106,69],
        #                    [104,86],[103,104],[101,119],[100,134],[100,148],[106,162],[113,170],[169,136],[157,151],[156,169],
        #                    [161,177],[160,184],[150,178],[144,166],[132,164],[123,153],[114,143],[115,130],[118,118],[129,111],
        #                    [120,42],[151,109],[109,105],[116,90],[123,82],[122,70]], dtype=np.int64)


        #print(self.net)
        #summary(self.net, (1, 3, 256, 256))


    
    def load_model(self):
        self.model = torch.load(self.model_path)
        #self.model.eval()

    def root_mean_squared_error(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_pred - y_true)))

    
    def init_transform_ht(self):
        return transforms.Compose([
                    ToHeatmap()
                ])

    
    def align_affine(self, img, pts, idx, folder, og, align_type='affine'):
        #inter = {0: cv2.INTER_NEAREST, 1: cv2.INTER_LINEAR, 2: cv2.INTER_AREA,
        #     3: cv2.INTER_CUBIC, 4: cv2.INTER_LANCZOS4, 5: cv2.INTER_LANCZOS4}
        #border = {'constant': cv2.BORDER_CONSTANT, 'edge': cv2.BORDER_REPLICATE,
        #      'symmetric': cv2.BORDER_REFLECT, 'reflect': cv2.BORDER_REFLECT101,
        #      'wrap': cv2.BORDER_WRAP}
        ear_factor = 0.75
        crop_size = 256
        img = np.transpose(img,(1,2,0))
        #og = np.transpose(og,(1,2,0))

        #print(pts.shape)
        #print(self.standard_lds.shape)

        #if idx == 3:
        #    print(pts)

        #plt.figure(frameon=False)
        #print(img)
        #plt.imshow((img*255).astype(np.uint8)) #(img* 255).astype(np.uint8))
        #plt.scatter(self.standard_lds[:, 0], self.standard_lds[:, 1], s=20, marker='.', c='r')
        #plt.scatter(pts[:, 0], pts[:, 1], s=20, marker='.', c='b')
        #plt.scatter(pts[:, 0], pts[:, 1], s=20, marker='.', c='g')
        #plt.show()
        print(folder)
        folder_ = folder[0].split('/')[2]
        img_folder = folder[0].split('/')[3]

        #cv.imwrite('data/aligned_awe/'+folder+'/'+str(idx)+'_og.png', cv.cvtColor((img*255).astype(np.uint8), cv.COLOR_BGR2RGB))

        def scale_data(data, max_n, min_n):
            max_data = np.max(data)
            min_data = np.min(data)
            return (max_n - min_n)/(max_data-min_data)*(data - max_data) + max_n

        trg_landmarks = scale_data(self.standard_lds, 1, 0) * crop_size * ear_factor + np.array([crop_size // 8, (crop_size // 8)])
        #pts = scale_data(pts, 1, 0) * crop_size * ear_factor + np.array([crop_size // 2, -(crop_size // 2)])

        # estimate transform matx
        if align_type == 'affine':
            tform = cv.estimateAffine2D(trg_landmarks, pts, ransacReprojThreshold=np.Inf)[0]
        else:
            # similarity
            tform = cv.estimateAffinePartial2D(trg_landmarks, pts, ransacReprojThreshold=np.Inf)[0]

        # warp image by given transform
        
        #print(img.shape)
        #print(img)
        img = og
        aligned_img = cv.warpAffine(img.astype(np.uint8), tform, (crop_size, crop_size), flags=cv.WARP_INVERSE_MAP + cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

        # get transformed landmarks
        tformed_landmarks = cv.transform(np.expand_dims(pts, axis=0), cv.invertAffineTransform(tform))[0]

        """plt.figure(frameon=False)
        plt.imshow(aligned_img.astype(np.uint8))
        plt.axis('off')
        #plt.savefig('aligned_awe/'+folder+'/' + str(idx) + '.png', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.show()
        plt.pause(3)
        plt.close()"""

        def calc_img_bbox(points):
            # crop images based on landmarks TODO:(=probably needs to be done)
            # define bbox by landmarks max/min
            ld_x = points[:,0]
            ld_y = points[:,1]
            bbox = (ld_x.min(), ld_y.min(), ld_x.max(), ld_y.max())

            return bbox

        def check_val(val, pad):
            if val < 0:
                val = 0
            if (val - pad) < 0:
                pad = 5
            if (val - pad) < 0:
                pad = 0
            return pad, val

        def check_val2(val, pad, h):
            if val >= h:
                val = (h-1)
            if (val + pad) >= h:
                pad = 5
            if (val + pad) >= h:
                pad = 0
            return pad, val

        bbox = calc_img_bbox(tformed_landmarks)
        print(bbox)
        sx, sy, w, h = bbox
        pad = 5
        pax = 5
        pay = 5
        pas = 5

        pas, sx = check_val(sx, pas)
        pay, sy = check_val(sy, pay)
        pad, h = check_val2(h, pad, aligned_img.shape[0])
        pax, w = check_val2(w, pax, aligned_img.shape[1])
        #print(sx, sy, w, h)
        aligned_crop = aligned_img[sy-pay:h+pad,sx-pas:w+pax]

        bbox_og = calc_img_bbox(pts)
        print(bbox_og)
        sx, sy, w, h = bbox_og
        pas, sx = check_val(sx, pas)
        pay, sy = check_val(sy, pay)
        pad, h = check_val2(h, pad, aligned_img.shape[0])
        pax, w = check_val2(w, pax, aligned_img.shape[1])
        og_crop = img[sy-pay:h+pad,sx-pas:w+pax]

        #og_pts_x = pts[:, 0] - pas
        #og_pts_y = pts[:,1] - pay

        #pts[:,0] = og_pts_x
        #pts[:,1] = og_pts_y


        #og = cv.cvtColor((og*255), cv.COLOR_BGR2RGB)
         

        """plt.figure(frameon=False)
        plt.imshow((og).astype(np.uint8))
        plt.axis('off')
        plt.savefig('data/og_awe_ext_lds/train_joint/'+folder_+'/' + img_folder, bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.show()
        #plt.pause(0.1)
        plt.close()"""


        """plt.figure(frameon=False)
        #print(img)
        plt.imshow((og).astype(np.uint8)) #(img* 255).astype(np.uint8))
        #plt.scatter(self.standard_lds[:, 0], self.standard_lds[:, 1], s=20, marker='.', c='r')
        #plt.scatter(tformed_landmarks[:, 0], tformed_landmarks[:, 1], s=20, marker='.', c='b')
        plt.scatter(pts[:, 0], pts[:, 1], s=20, marker='.', c='g')
        #plt.show()
        #plt.pause(1)
        plt.axis('off')
        #plt.savefig('data/aligned_awe/' + str(idx) + '_landmarks.png', bbox_inches='tight', pad_inches=0, transparent=True)
        #plt.close()"""

        # write landmark pts
        """with open('data/og_awe_ext_lds/train_joint/'+folder_+'/' + img_folder.split('.')[0] + '.pts', 'w') as file:
            file.write('version: 1\n')
            #print('n_points: '+str(tformed_landmarks.shape[0]))
            file.write('n_points: '+str(pts.shape[0])+'\n')
            file.write('{\n')
            for pt in pts:
                #print(pt)
                file.write(str(pt[0])+' '+str(pt[1])+'\n')
            file.write('}')"""

        return aligned_img, tformed_landmarks


    
    def align_ear(self, img, pts):
        X = pts

        cv.imshow('img', np.asarray(img*255).astype(np.uint8))
        #cv.waitKey(0)
        #cv.imshow('x', (X*255).astype(np.uint8))
        #cv.waitKey(0)
        # Perform a PCA and compute the angle of the first principal axes
        pca = PCA(n_components=2).fit(X)
        angle1 = np.arctan2(*pca.components_[0])
        angle2 = np.arctan2(*pca.components_[1])
        print("=angles=")
        print(angle1)
        print(angle2)

        row,col = img.shape
        center=tuple(np.array([row,col])/2)
        rot_mat = cv.getRotationMatrix2D(center,angle2,1.0)
        print(rot_mat)
        rot_mat2 = cv.getRotationMatrix2D(center,angle2,1.0)
        #rot_mat = np.matmul(shear_mat, rot_mat)
        #print(rot_mat)

        #print(rot_mat)
        new_image = cv.warpAffine(np.asarray(img), rot_mat, (col,row))
        new_image = cv.warpAffine(np.asarray(new_image), rot_mat2, (col,row))

        #sh_mat = np.array([[1,angle1,0],[angle1,1,0],[0,0,1]], dtype=np.float32)
        #sheared = cv.warpPerspective(new_image, sh_mat, (col,row))
        #n#ew_image = AffineTransform(rot_mat, )
        #angle3 = np.arctan2(*pca.components_[2])
        # Rotate the image by the computed angle:
        #rotated_img = rotate(np.asarray(img),(angle2/np.pi*(-180))+90)
        cv.imshow('rot (1,0)', new_image)
        #rotated_img = rotate(rotated_img,angle2/np.pi*180-90, axes=(2,1))
        #cv.imshow('rot (1,2)', sheared)
        #rotated_img = rotate(rotated_img,angle2/np.pi*180-90, axes=(2,0))
        #cv.imshow('rot (2,0)', rotated_img)
        cv.waitKey(0)
        return new_image


    
    def predict_keypoints(self):
        # organize the dataset: {'image':...,'landmarks':...}
        dataset = self.dataset
        #print(dataset)
        vs = self.vs
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        #print("device: ", device)

        test_loader = DataLoader(dataset, 
                                batch_size=1,
                                shuffle=False, 
                                num_workers=0)

        print(len(test_loader))

        lds = np.zeros((len(test_loader), 55, 2))
        gt_lds_all = np.zeros((len(test_loader), 55, 2))
        scores = []
        mse = []
        cts = []
        out_imgs = []
        times=[]
        mae=[]
        avg_lds = np.zeros((55,2))

        # iterate through the test dataset
        for i, sample in enumerate(test_loader):
            # get sample data: images and ground truth landmarks
            img = sample['image']
            #print(img.shape)
            folder = sample['landmarks']
            og = sample['original']
            #print(folder)
            #print(gt_lds)
            #gt_lds = gt_lds.data.numpy()[0]
            #print(gt_lds.shape)
            #print(gt_lds[54])
            #print(gt_lds[55])
            
            # convert images to FloatTensors
            image_tensor = img.type(torch.FloatTensor)

            # forward pass to get net output
            t1 = time.time()
            output_pts = self.net(image_tensor)
            print("prediction")

            ##GET KEYPOINTS

            output_pts0 = output_pts[0].data.numpy()
            #output_pts1 = output_pts[0].data.numpy()
            #print(output_pts.shape)

            """output_pts_ = np.zeros((64,64))
            output_pts_1 = np.zeros((64,64)) 
            
            for idx,j in enumerate(output_pts1):
                print(j.shape)
                #j = j[0].data.numpy()
                #output_pts_ += j
                img = (255*(j - np.min(j))/np.ptp(j)).astype(np.uint8)  
                cv.imwrite('out_feature1_' + str(i) + '_+'+str(idx)+'.png', img)"""
            #print(output_pts.shape)
            predicted_pts = []
            #gt_pts = []
            for j, points in enumerate(output_pts):
                h, w = np.unravel_index(points.argmax(), points.shape)
                x = int(w * 256 / points.shape[1])
                y = int(h * 256 / points.shape[0])
                predicted_pts.append([x,y])

                #gt = gt_lds[j]
                #print(gt)

                #h, w = np.unravel_index(gt.argmax(), gt.shape)
                #x = gt[0]#int(w * 256 / gt.shape[1])
                #y #= gt[1]#int(h * 256 / gt.shape[0])
                #gt_pts.append([x,y])

            t2 = (time.time() - t1)
            #print("FPS: ", t2)

            times.append(t2)

            predicted_pts = np.asarray(predicted_pts)
            lds[i] = predicted_pts

            #gt_pts = np.asarray(gt_pts)
            #gt_lds_all[i] = gt_pts

            

            #en_mse = mean_squared_error(gt_pts, predicted_pts)
            #en_mae = mean_absolute_error(gt_pts, predicted_pts)

            #print("MSE", en_mse)
            #print("MAE", en_mae)
            
            #vs.show_landmarks(img, predicted_pts, i, gt_lds=gt_pts)
            _, ldk = self.align_affine(img.squeeze(0).data.numpy(), predicted_pts, i, folder, og.squeeze(0).data.numpy())
            avg_lds = avg_lds + ldk
            
            

            #unaligned_rgb = img.squeeze(0)
            #print(unaligned_rgb.shape)
            #rotated_r = self.align_ear(unaligned_rgb[0,:,:], predicted_pts)
            #rotated_b = self.align_ear(unaligned_rgb[2,:,:], predicted_pts)
            #rotated_g = self.align_ear(unaligned_rgb[1,:,:], predicted_pts)
            #aligned_rgb = np.zeros((256,256,3))
            #aligned_rgb[:,:,0] = rotated_b
            #aligned_rgb[:,:,1] = rotated_g
            #aligned_rgb[:,:,2] = rotated_r

            #cv.imwrite('data/aligned/'+str(i)+'.png', aligned_rgb)
            #cv.imshow('rot', aligned_rgb)
            #cv.waitKey(0)

            # this is where we get all landmarks

            """tf = self.init_transform_ht()
            sample = tf({'image': img[0], 'landmarks': torch.from_numpy(gt_pts)})
            _, label = sample['landmarks']
            fg = label * 255

            sample2 = tf({'image': img[0], 'landmarks': torch.from_numpy(predicted_pts)})
            _, label2 = sample2['landmarks']
            label = label2 * 255
            cv.imwrite('data/sample/label_'+str(i)+'.png', (label).astype(np.uint8))

            output_array_ = (fg - np.min(fg)) / (np.max(fg) - np.min(fg))
            label_array = (label - np.min(label)) / (np.max(label) - np.min(label))        

            intersection = np.logical_and(label_array, output_array_)
            union = np.logical_or(label_array, output_array_)
            iou_score = np.sum(intersection) / np.sum(union)
            #print('IOU: ', iou_score)
            scores.append(iou_score)#"""

            #mse.append(en_mse)
            #mae.append(en_mae)

            cts.append(i)
            print("=ct=", i)

        #print(gt_lds_all.shape)
        #print("=this is baseline=")
        #print(avg_lds.shape)
        avg_lds = avg_lds / (cts[-1]+1)
        with open('data/og_awe_ext_lds/avg_lds.pts', 'w') as file:
            file.write('version: 1\n')
            #print('n_points: '+str(tformed_landmarks.shape[0]))
            file.write('n_points: '+str(avg_lds.shape[0])+'\n')
            file.write('{\n')
            for pt in avg_lds:
                #print(pt)
                file.write(str(pt[0])+' '+str(pt[1])+'\n')
            file.write('}')

        plt.figure()
        plt.scatter(avg_lds[:,0], avg_lds[:,1])
        plt.show()

        #repeated_avg_lds = np.repeat([avg_lds], cts[-1]+1, axis=0)
        #print(repeated_avg_lds.shape)

        plt.plot(self.iter_list, self.train_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Normalized MSE loss')
        plt.grid(True)
        plt.legend(('2-SHGNet train loss'), loc='lower right')
        plt.savefig('data/result_figures/train_loss_2shgnet.png', bbox_inches='tight')
        plt.close()

        plt.plot(cts, times)
        plt.xlabel('Test samples')
        plt.ylabel('Time (s)')
        #plt.title('FPS value over test samples')
        plt.grid(True)
        plt.legend(('Prediction time for 2-SHGNet'), loc='lower right')
        plt.savefig('data/result_figures/fps_hg_2.png', bbox_inches='tight')
        plt.close()

        avg_mae = []
        avg_mse = []

        for idx,val in enumerate(gt_lds_all):
            gts = gt_lds_all[idx]
            #print(gts)
            #print(gts.shape)
            avgs = avg_lds
            avg_mae.append(mean_absolute_error(gts, avgs))
            avg_mse.append(mean_squared_error(gts, avgs))
        # axis=1  MSE za landmarke za posamezno sliko
        #print('MSE - by test sample: ', ((gt_lds_all - lds)**2).mean(axis=1).mean())

        def scale_data(data, max_n, min_n):
            max_data = np.max(data)
            min_data = np.min(data)
            return (max_n - min_n)/(max_data-min_data)*(data - max_data) + max_n

        proportions = scale_data(cts, 100, 0)

        # evaluate the histogram
        values, base = np.histogram(mse, bins=100)
        values_avg, base_mse = np.histogram(avg_mse, bins=100)
        #evaluate the cumulative
        norm_mse = np.cumsum(values)
        norm_avg_mse = np.cumsum(values_avg)

        proportions = scale_data(norm_mse, 100, 0)
        props2 = scale_data(norm_avg_mse,100,0)

        """plt.plot(base[:-1],proportions)
        plt.plot(base_mse[:-1],props2)
        plt.xlabel('Normalized Cumulative MSE')
        plt.ylabel('Proportion of test samples')
        #plt.title('FPS value over test samples')
        plt.grid(True)
        plt.legend(('2-SHGNet cumulative MSE', 'Baseline cumulative MSE'), loc='lower right')
        #plt.savefig('data/result_figures/CMSE_2shgnet.png', bbox_inches='tight')
        plt.show()

        # evaluate the histogram
        values, base = np.histogram(mae, bins=100)
        values_avg, base_mae = np.histogram(avg_mae, bins=100)      
        norm_mae = np.cumsum(values)
        norm_avg_mae = np.cumsum(values_avg)  

        print("2stack MSE", np.average(mse))
        print("2stack MAE", np.average(mae))
        print("avg MSE", np.average(avg_mse))
        print("avg MAE", np.average(avg_mae))

        proportions = scale_data(norm_mae, 100, 0)
        props2 = scale_data(norm_avg_mae,100,0)
        plt.plot(base[:-1], proportions)
        plt.plot(base_mae[:-1],props2)
        plt.xlabel('Normalized Cumulative MAE')
        plt.ylabel('Proportion of test samples')
        #plt.title('FPS value over test samples')
        plt.grid(True)
        plt.legend(('2-SHGNet cumulative MAE', 'Baseline cumulative MAE'), loc='lower right')
        #plt.savefig('data/result_figures/CMAE_2shgnet.png', bbox_inches='tight')
        plt.show()"""

        """scores = np.cumsum(np.asarray(scores))
        plt.plot(cts, scores)
        plt.xlabel('Test samples')
        plt.ylabel('Cumulative Intersection over union (IOU)')
        plt.title('Cumulative IOU score over test samples')
        plt.savefig('data/result_figures/cum_iou_hg.png', bbox_inches='tight')
        plt.close()"""

        #print(mse)
        #cumulative_mse = np.cumsum(np.asarray(mse)) / len(mse)

        """plt.plot(cumulative_mse, cts)
        plt.ylabel('Test samples')
        plt.xlabel('Cumulative MSE Loss')
        plt.title('Cumulative MSE Loss over test samples')
        plt.savefig('data/result_figures/cum_mse_samp_reversed2_hg.png', bbox_inches='tight')
        plt.close()"""

        # axis=0  Mza specific landmark
        mse_ld = np.mean((gt_pts.T - predicted_pts.T)**2, axis=0) #/ np.mean(gt_pts.T)
        cumulative_mse = list(np.cumsum(np.asarray(mse_ld)))
        cts_ = [i for i in range(len(cumulative_mse))]
        #cts = (cts - np.min(cts)) / (np.max(cts) - np.min(cts))
        """plt.plot(cumulative_mse, cts_)
        plt.xlabel('Cumulative MSE Loss')
        plt.ylabel('Landmark class')
        plt.title('Cumulative MSE Loss over landmark classes')
        plt.savefig('data/result_figures/cum_mseldnew_hg.png', bbox_inches='tight')
        plt.close()"""

        plt.plot(cts_, cumulative_mse)
        plt.ylabel('Cumulative MSE Loss')
        plt.xlabel('Landmark class')
        plt.title('Cumulative MSE Loss over landmark classes')
        #plt.savefig('data/result_figures/cum_mseld_class_hg.png', bbox_inches='tight')
        plt.close()

        print("2stack MSE", np.average(mse))
        print("2stack MAE", np.average(mae))
        print("avg MSE", np.average(avg_mse))
        print("avg MAE", np.average(avg_mae))

        """print("R2 score: ", r2_score(gt_lds_all, lds) * 100)"""

        # train loss
        train_loss = self.train_loss
        iters = self.iter_list

        #train_loss = (train_loss - np.min(train_loss)) / (np.max(train_loss) - np.min(train_loss))

        """plt.plot(iters, train_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Normalized MSE train loss')
        plt.title('Normalized MSE train loss over epoch')
        plt.savefig('data/result_figures/train_loss_hg_awe.png', bbox_inches='tight')
        plt.close()"""

        # test loss
        """test_loss = mse
        test_loss = np.cumsum(np.asarray(test_loss))

        plt.plot(cts,test_loss)
        plt.xlabel('Testing iteration')
        plt.ylabel('MSE test loss')
        plt.title('MSE test loss over testing iteration')
        plt.savefig('data/result_figures/test_loss_mse_hg_awe.png', bbox_inches='tight')
        plt.close()"""


        #writer.add_graph(self.net, out_imgs)
        #writer.close()
        #test_loss = (test_loss - np.min(test_loss)) / (np.max(test_loss) - np.min(test_loss))
        #print(test_loss)
        #plt.plot(cts, test_loss)
        #plt.xlabel('Testing iteration')
        #plt.ylabel('Normalized MSE test loss')
        #plt.title('Normalized MSE test loss over testing iteration')
        #plt.savefig('data/result_figures/test_loss_hg.png', bbox_inches='tight')
        #plt.close()




        
        
        
        # reshape to batch_size x 68 x 2 pts
        #output_pts = output_pts.view(output_pts.size()[0], 55, -1)
        #print("", output_pts.shape)
        #output_pts_unscaled = output_pts.data.numpy()[0]*50+100
        #print("unscaled ", output_pts_unscaled.shape)
        #if lds.all() == None:
        #    lds = output_pts_unscaled
        #else:
        #    lds = np.concatenate((lds, output_pts_unscaled), axis=0)
        #print("all ", lds.shape)
        #images = np.append(images, img)
        
        # break after first image is tested
        #if i == 0:
        #yield(images, lds, gt_lds)
        #print(img)
        #if self.test:
        #    vs.show_landmarks(img, output_pts, i, gt_lds=gt_lds)

        #mse = mean_squared_error(gt_lds_all, lds)
        """r_square = r2_score(gt_lds_all, lds)
        # axis=1  MSE za landmarke za posamezno sliko
        print('MSE - by test sample: ', ((gt_lds_all - lds)**2).mean(axis=1).mean())
        print('RMSE: ', self.root_mean_squared_error(gt_lds_all, lds))

        # axis=0  MSE za specific landmark
        print('MSE - by landmark: ' , ((gt_lds_all - lds)**2).mean(axis=0).mean())

        print("R2 score: ", r2_score(gt_lds_all, lds) * 100)
        """

        """cnt = 0
        for i in self.images:
            img = self.images[i]
            img_tensor = self.dataset[cnt]['image']
            #print(img.shape)
            img_tensor = img_tensor[None, :]

            #print(img.shape)

            # dtype('uint8')
            #img = torch.from_numpy(img)

            prediction = self.net(img_tensor.float())
            #print(len(prediction[0]))
            #print(prediction)
            #print(prediction.shape)

            #print(img)

            for landmark in range(len(prediction[0])):
                # adjust the landmark position because of rescaling 
                prediction[0][landmark] = int(prediction[0][landmark] * 224)

            self.draw_landmarks(prediction[0], i, cnt)

            cnt += 1"""
