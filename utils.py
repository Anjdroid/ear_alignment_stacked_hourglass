import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
from torchvision import transforms, utils
#import open3d
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import sys
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import random
import math

np.set_printoptions(threshold=sys.maxsize)

#BEST_LDS = [35,54,53,52,51,50,49,48,8,9,10,11,12,47,46,45,44,43,42,41,20,21,22,23,40,38,37,36,28,29,30,31,32,33,34,27,7,26,13,39,24,25,14,6,5,15,2,0,16,17,18,4,3,19,1]
#BEST_LDS = [33, 54, 53, 52, 51, 50, 49, 48, 47, 9, 10, 11, 46, 45, 44, 43, 42, 41, 40, 39, 20, 21, 22, 23, 38, 37, 36, 35, 28, 34, 30, 31, 32, 27, 29, 26, 12, 25, 8, 15, 7, 13, 14, 16, 1, 17, 18, 19, 24, 6, 5, 4, 3, 2, 0]

BEST_LDS = [0,2,4,7,8,10,12,13,14,16,18,19,20,24,26,28,30,32,34,36,37,38,39,42,44,45,46,47,48,49,50,52,54]

MAX_RES = 256



# tranforms

class Equalize(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image_copy = np.copy(image)
        lds_copy = np.copy(landmarks)

        img_ycr = cv.cvtColor(image_copy, cv.COLOR_BGR2YCR_CB)
        channels = cv.split(img_ycr)
        #print(channels)
        cv.equalizeHist(channels[0],channels[0])
        cv.merge(channels, img_ycr)
        cv.cvtColor(img_ycr, cv.COLOR_YCR_CB2BGR, image_copy)

        #cv.imshow('',image_copy)
        #cv.waitKey(0)

        return {'image': image_copy, 'landmarks': lds_copy}

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        #print("norm ", sample)
        
        image_copy = np.copy(image)
        lds_copy = np.copy(landmarks)

        #print(image_copy)

        # convert image to grayscale
        #image_copy = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

        #print(image_copy.shape)
        image_r = image_copy[:,:,0] / 255.0
        image_g = image_copy[:,:,1] / 255.0
        image_b = image_copy[:,:,2] / 255.0

        #cv.imshow('img', image_copy)
        #cv.waitKey(0)
        #print(image_copy)
        
        # scale color range from [0, 255] to [0, 1]
        #norm_img =  image_copy/255.0
        norm_img = np.zeros(image_copy.shape)
        #print(norm_img)
        norm_img[:,:,0] = image_r
        norm_img[:,:,1] = image_g
        norm_img[:,:,2] = image_b        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        #lds_copy = (lds_copy - 100)/50.0

        return {'image': norm_img, 'landmarks': lds_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        #print("rescale, ", sample)
        image, landmarks = sample['image'], sample['landmarks']

        #print(image.shape)
        #print(landmarks)

        h, w, _ = image.shape
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv.resize(image, (new_w, new_h))
        #print("img ", img)
        
        # scale the pts, too
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}

class Augment(object): 
    # https://imgaug.readthedocs.io/en/latest/source/examples_heatmaps.html
    def __init__(self, num_aug, lds, num_lds):
        #ia.seed(1)
        self.num_aug = num_aug
        self.lds = lds
        self.num_lds = num_lds

        sometimes = lambda aug: iaa.Sometimes(1.0, aug)
        self.seq = iaa.Sequential(
            # Define our sequence of augmentation steps that will be applied to every image.
            [
            # Apply the following augmenters to most images.
            #
            #iaa.Fliplr(0.5), # horizontally flip 50% of all images
            #iaa.Flipud(0.5), # vertically flip 20% of all images

            # Apply affine transformations to some of the images
            # - scale to 80-120% of image height/width (each axis independently)
            # - translate by -20 to +20 relative to height/width (per axis)
            # - rotate by -45 to +45 degrees
            # - shear by -16 to +16 degrees
            # - order: use nearest neighbour or bilinear interpolation (fast)
            # - mode: use any available mode to fill newly created pixels
            #         see API or scikit-image for which modes are available
            # - cval: if the mode is constant, then use a random brightness
            #         for the newly created pixels (e.g. sometimes black,
            #         sometimes white)
            sometimes(iaa.Affine(
                scale={"x": (0.7, 0.95), "y": (0.7, 0.95)},
                #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-45, 45),
                shear=(-16, 16),
                order=[0, 1],
                #cval=(255),
                mode='edge'
            )),

            #sometimes(iaa.GaussianBlur(sigma=(0, 0.5))),
            #random_order=True
            ])

    def __call__(self, sample):

        ia.seed(1)

        # heatmap must be bounded 0.0-1.0
        # heatmaps will ONLY be affected by augmentations that change the geometry of images 
        
        # samples have to be numpy arrays (before tensors)
        # INPUT: sample['image'], sample['landmarks'] -- all htmps lds

        image = sample['image'] 
        mask = sample['landmarks']

        #print(image)
        #print(mask)
        #print(mask)
        # get htmps

        """ht_pts = []

        #best_lds = BEST_LDS[::]
        #best_lds = sorted(best_lds)
        best_lds = [i for i  in range(55)]
        #pri-nt("best lds")
        #print(BEST_LDS)
        #print(len(BEST_LDS))
        #print(len(best_lds))
        #print(best_lds)
                   
        #cnt = 0
        #best_ld = 0
        #for idx in range(0, len(landmarks), 2):
        #    #print(len(landmarks))
        #    #print(idx)
        #    #print(landmarks[idx])
        #    if cnt >= 55:
        #        break
        #    elif idx == best_lds[0]:

        #ht = np.zeros((256,256))
        best_idx = 0
        for i in range(55):
            if i == best_lds[best_idx]:
                ht = cv.cvtColor(cv.imread(self.data_path + mask.split('.')[0] + '.png_' + str(i) + '.jpg'), cv.COLOR_BGR2GRAY)
                h, w = np.unravel_index(ht.argmax(), ht.shape)
                ht_x = int(w * 256 / ht.shape[1])
                ht_y = int(h * 256 / ht.shape[0])
                ht_pts.append([ht_x,ht_y])
                best_idx += 1
        #print(len(ht_pts))"""

        #if mask.any() != None:
        kps_lst = np.array([Keypoint(x=x, y=y) for (x,y) in mask])
        keypoints = KeypointsOnImage(kps_lst, shape=image.shape)
        num_aug = self.num_aug
        images_aug = []
        heatmaps_aug = []
        for i in range(num_aug):

            #if mask.any() != None:
            images_aug_i, heatmaps_aug_i = self.seq(image=image, keypoints=keypoints)
            images_aug.append(images_aug_i)
            heatmaps_aug.append([i.xy for i in heatmaps_aug_i])
            cv.imwrite('data/sample/img_'+str(i)+'.png', cv.cvtColor((images_aug_i).astype(np.uint8), cv.COLOR_BGR2RGB))
            #cv.imwrite('data/sample/heatmap_'+str(i)+'.png', images_aug_i)
            #else:
            #    images_aug_i = self.seq(image=image)
            #    images_aug.append(images_aug_i)
            #cv.imshow('',images_aug_i)
            #cv.waitKey(0)

        images_aug.append(image)

        #if mask.any() != None:
        #heatmaps_aug.append([i.xy for i in kps_lst])
        #kp_x = np.asarray(heatmaps_aug[0]).reshape(self.num_lds,2)[:,0]
        #kp_y = np.asarray(heatmaps_aug[0]).reshape(self.num_lds,2)[:,1]
        #print(heatmaps_aug[-1])

        img = images_aug[0]
        


        #plt.figure()
        #plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)) #(img* 255).astype(np.uint8))
        ##plt.scatter(heatmaps_aug[-1,0,:], heatmaps_aug[-1,1,:], s=20, marker='.', c='b')
        #plt.scatter(kp_x, kp_y, s=20, marker='.', c='b')
        #plt.pause(15)

        # print coordinates before/after augmentation (see below)
        # use after.x_int and after.y_int to get rounded integer coordinates
        """for j in range(num_aug):
            for i in range(len(keypoints.keypoints)):
                before = keypoints.keypoints[i]
                after = heatmaps_aug[j].keypoints[i]
                print("Keypoint %d: (%.8f, %.8f) -> (%.8f, %.8f)" % (
                    i, before.x, before.y, after.x, after.y)
                )
            
            # image with keypoints before/after augmentation (shown below)
            #image_before = keypoints.draw_on_image(image, size=3)
            #image_after = heatmaps_aug[j].draw_on_image(images_aug[j], size=3, color=255)
            #htmp = np.zeros((image.shape[0],image.shape[1]))
            #htmp = heatmaps_aug[j].draw_on_image(htmp, size=3, color=255)
            #print(htmp.shape)

            #numpy_horizontal_concat = np.concatenate((image_before, image_after), axis=1)
            #numpy_horizontal_concat = np.concatenate((numpy_horizontal_concat, htmp), axis=1)
            #cv.imshow('', numpy_horizontal_concat)
            #cv.waitKey(0)"""
        return {'image': images_aug, 'landmarks': heatmaps_aug}



class Crop(object):
    """Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, bbox):
        self.bbox = bbox

    
    def add_padding_min(self, val, m):
        if (val - m) > 0:
            val = val - m
        elif val < 0:
            val = 0
        else:
            val = (val % m)
        return int(val)


    def add_padding_max(self, val, max_val, m):
        if (val + m) < max_val:
            val = val + m
        else:
            val = max_val
        return int(val)


    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h_img, w_img, _ = image.shape

        """if landmarks.any() == None:
            sx, sy = 0, 0
            gx = w_img
            gy = h_img
            cropped_ear = image[sy:gy,sx:gx]
            h, w, _ = cropped_ear.shape
        else:"""
        sx, sy, gx, gy = self.bbox        
        rand_x = np.random.randint(20, 40)
        # add padding
        sx = self.add_padding_min(sx, rand_x)
        rand_y = np.random.randint(20,40)
        sy = self.add_padding_min(sy, rand_y)
        # add padding
        gx = self.add_padding_max(gx, w_img, rand_x)
        gy = self.add_padding_max(gy, h_img, rand_y)
        cropped_ear = image[sy:gy,sx:gx]
        h, w, _ = cropped_ear.shape

        if h == w:
            cpy = cropped_ear.copy()
            res = h - w
            # res is 0
            resized = cv.resize(cpy, ((MAX_RES, MAX_RES)))      
            res_x1 = 0  
        elif h > w:
            cpy = np.zeros((h, h, 3))
            res = h - w
            res_x1 = math.ceil(res/2)
            res_x2 = math.floor(res/2)
            if (sx - res_x1) < 0:
                temp = res_x1
                res_x1 = sx
                res_x2 = res_x2 + (temp - sx)
            if (gx + res_x2) >= w_img:
                temp = res_x2
                res_x2 = (w_img - gx)
                res_x1 = res_x1 + (temp - res_x2)

            if landmarks == None:
                res_x1 = math.ceil(res/2)
                res_x2 = math.floor(res/2)
                replicate = cv.copyMakeBorder(cropped_ear,0,0,res_x1,res_x2,cv.BORDER_REPLICATE)

                resized = cv.resize(replicate, ((MAX_RES, MAX_RES))) 

            else:
                cpy = image[sy:gy, (sx - res_x1):sx]
                cpy2 = image[sy:gy, gx:(gx + res_x2)]

                merged = np.zeros((h, h, 3))
                merged[:,:res_x1] = cpy[:,:,:]
                merged[:,(h-res_x2):] = cpy2[:,:,:]
                merged[:,res_x1:(h-res_x2)] = cropped_ear
                
                h, w, _ = merged.shape
                resized = cv.resize(merged, ((MAX_RES, MAX_RES)))            
        else:
            cpy = np.zeros((w, w, 3))
            res = w - h

            # h == 256
            #box_img = np.zeros((MAX_RES, MAX_RES, 3))
            res_x1 = math.ceil(res/2)
            res_x2 = math.floor(res/2)

            if (sy - res_x1) < 0:
                temp = res_x1
                res_x1 = sy
                res_x2 = res_x2 + (temp - sy)
            
            elif (gy + res_x2) >= h_img:
                temp = res_x2
                res_x2 = (h_img - gy)
                res_x1 = res_x1 + (temp - res_x2)

            if landmarks == None:
                res_x1 = math.ceil(res/2)
                res_x2 = math.floor(res/2)
                replicate = cv.copyMakeBorder(cropped_ear,res_x1,res_x2,0,0,cv.BORDER_REPLICATE)
                resized = cv.resize(replicate, ((MAX_RES, MAX_RES)))

            else:

                #left = np.zeros((MAX_RES, res_x1))
                cpy = image[(sy-res_x1):sy, sx:gx]
                #cpy = cv.resize(cpy, (h,h))
                cpy2 = image[gy:(gy+res_x2), sx:gx]

                merged = np.zeros((w, w, 3))
                merged[:res_x1,:] = cpy[:,:,:]
                merged[(w-res_x2):,:] = cpy2[:,:,:]
                merged[res_x1:(w-res_x2),:] = cropped_ear
                
                #box_img[:,int(res_x/2):int(res_x + (res_x/2))] = resized
                h, w, _ = merged.shape
                resized = cv.resize(merged, ((MAX_RES, MAX_RES)))

        if landmarks != None:
            # center to ear region
            landmarks = landmarks - [sx, sy]
            #print(merged.shape)

            landmarks = landmarks + [res_x1, 0]

            # scale base on new h,w
            landmarks = landmarks * [resized.shape[0] / w, resized.shape[1] / h]

        #plt.figure()
        #plt.imshow(resized.astype(np.uint8))
        #plt.scatter(landmarks[:,0], landmarks[:,1])
        #plt.show()
        return {'image': resized.astype(np.uint8), 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        #print("tensotr, ", sample)
        image, landmarks = sample['image'], sample['landmarks']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # opencv image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))       
        
        return {'image': torch.from_numpy(image),
                'landmarks': []}


class ToHeatmap(object):
    """ converts landmarks to heatmaps """

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        #print(landmarks)
        #print(torch.unsqueeze(landmarks, 0)) 
        #print(image.shape)

        return {'image': image,
                'landmarks': self.generate_hm(image.shape[1], image.shape[2], torch.unsqueeze(landmarks, 0), 2)}
        #print(landmarks)
        #print(hm)

    def gaussian_k(self, x0,y0,sigma, height, width):
        """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
        """
        x = np.arange(0, width, 1) ## (width,)
        y = np.arange(0, height, 1)[:, np.newaxis] ## (height,1)

        #print(np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2)))
        return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))

    def generate_hm(self, height, width ,landmarks, s):
        """ Generate a full Heap Map for every landmarks in an array
        Args:
            height    : The height of Heat Map (the height of target output)
            width     : The width  of Heat Map (the width of target output)
            joints    : [(x1,y1),(x2,y2)...] containing landmarks
            maxlenght : Lenght of the Bounding Box
        """
        Nlandmarks = len(landmarks[0])
        #print(Nlandmarks)
        hms = np.zeros((Nlandmarks, height, width))
        hmm_ = np.zeros((height, width))
        
        for i in range(Nlandmarks):
            hm = np.zeros((height, width), dtype = np.float32)
            if not np.array_equal(landmarks[0][i], [-1,-1]):

                #print(landmarks[0][i][0])
                #print(landmarks[0][i][1])
                #print(hm)

                g = self.gaussian_k(int(landmarks[0][i][0]),
                                        int(landmarks[0][i][1]),
                                        int(s), int(height), int(width))

                #print(g)

                

                hm = np.add(hm, g)

                #print(hm)

                #break

                #print(hm[:,:,i])
                #cv.imwrite('test.jpg', hm)
                
            else:
                hm = np.zeros((height,width))

            hms[i] = hm
            hmm_ += hm
            cv.imwrite('data/sample/mask_'+str(i)+'.png', (hm*255).astype(np.uint8))
            #break
        #print(hms.shape)

        #x = np.argwhere(hmm_ > 0)
        #print(x)
        #y = np.argwhere(hmm_ == 0)
        #print(y)
        """res = np.zeros(hmm_.shape)
        #hmm_[x] = 0
        res[np.where(hmm_==0)] = 1

        hm = np.ones((height, width), dtype=np.float32)
        for idx, ld in enumerate(landmarks[0]):
            print(idx, ld)
            ld = ld.data.numpy()
            #if not np.array_equal(landmarks[0][i], [-1,-1]):
            # g = gaussian_k(int(ld[0]),int(ld[1]),2, int(height), int(width))
            hm[int(ld[1]),int(ld[0])] = 0"""
        
        return hms, hmm_



    """def gaussian_k(x0,y0,sigma, width, height):
        # Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
        
        x = np.arange(0, width, 1, float) ## (width,)
        y = np.arange(0, height, 1, float)[:, np.newaxis] ## (height,1)
        return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))

    def generate_hm(height, width ,landmarks,s=3):
        # Generate a full Heap Map for every landmarks in an array
        #Args:
        #    height    : The height of Heat Map (the height of target output)
        #    width     : The width  of Heat Map (the width of target output)
        #    joints    : [(x1,y1),(x2,y2)...] containing landmarks
        #    maxlenght : Lenght of the Bounding Box
        
        Nlandmarks = len(landmarks)
        hm = np.zeros((height, width, Nlandmarks), dtype = np.float32)
        for i in range(Nlandmarks):
            if not np.array_equal(landmarks[i], [-1,-1]):
            
                hm[:,:,i] = gaussian_k(landmarks[i][0],
                                        landmarks[i][1],
                                        s,height, width)
            else:
                hm[:,:,i] = np.zeros((height,width))
        return hm
        
    def get_y_as_heatmap(df,height,width, sigma):
        columns_lmxy = df.columns[:-1] ## the last column contains Image
        columns_lm = [] 
        for c in columns_lmxy:
            c = c[:-2]
            if c not in columns_lm:
                columns_lm.extend([c])
        
        y_train = []
        for i in range(df.shape[0]):
            landmarks = []
            for colnm in columns_lm:
                x = df[colnm + "_x"].iloc[i]
                y = df[colnm + "_y"].iloc[i]
                if np.isnan(x) or np.isnan(y):
                    x, y = -1, -1
                landmarks.append([x,y])
                
            y_train.append(generate_hm(height, width, landmarks, sigma))
        y_train = np.array(y_train)
        
        
        return(y_train,df[columns_lmxy],columns_lmxy)"""


# visalization

class Visualize:
    def __init__(self):
        print("visualization class")


    def show_landmarks(self, img, predicted_lds, idx, gt_lds=None, result_path='data/awe_stackhg/'):
        """
        displays a rgb image,
        its predicted landmarks and its ground truth landmarks (if provided)
        """
        image = img.data   # get the image from it's wrapper
        image = image.numpy()  # convert to numpy array from a Tensor
        img = np.transpose(image[0], (1, 2, 0)) 
        #print(img.shape)
        #if img.shape[2] == 3:
        #img = cv.cvtColor(img.astype(np.uint8), cv.COLOR_BGR2RGB)

        plt.figure(frameon=False)
        #print(img)
        plt.imshow(img) #(img* 255).astype(np.uint8))
        plt.scatter(gt_lds[:, 0], gt_lds[:, 1], s=20, marker='.', c='r')
        plt.scatter(predicted_lds[:, 0], predicted_lds[:, 1], s=20, marker='.', c='b')

        #plt.pause(0.001)
        plt.axis('off')
        #plt.savefig(result_path + str(idx) + '.png', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.show()

    
    def show_ptc(self, ptc, file):
        ptc = open3d.io.read_point_cloud(file)
        # shift + left clck to choose points
        """ http://www.open3d.org/docs/release/python_api/open3d.visualization.draw_geometries_with_vertex_selection.html#open3d.visualization.draw_geometries_with_vertex_selection"""
        open3d.visualization.draw_geometries_with_vertex_selection([ptc], width=1400, height=900) # Visualize the point cloud


    def show_3D_landmarks(self, lds):
        fig = pyplot.figure()
        ax = Axes3D(fig)
        
        x = lds[:,0]
        y = lds[:,1]
        z = lds[:,2]
        ax.scatter(x, y, z)

        for i,txt in enumerate(lds):
            ax.text(x[i],y[i],z[i],'%s' % (str(i)), size=10, zorder=1,color='k') 
        pyplot.show()


    # visualize the output
    # by default this shows a batch of 10 images
    def visualize_output(self,test_images, test_outputs, gt_pts=None, batch_size=10):
        
        plt.figure(figsize=(20,10))
        for i in range(batch_size):
            
            plt.subplot(2, int(batch_size/2), i+1)
            
            # un-transform the image data
            image = test_images[i]   # get the image from it's wrapper
            #print(image.shape)
            #image = image.numpy()   # convert to numpy array from a Tensor
            #image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image

            #print(image.shape)

            # un-transform the predicted key_pts data
            predicted_key_pts = test_outputs[i].data
            predicted_key_pts = predicted_key_pts.numpy()
            # undo normalization of keypoints  
            predicted_key_pts = predicted_key_pts*50.0+100
            
            # plot ground truth points for comparison, if they exist
            ground_truth_pts = None
            if gt_pts is not None:
                ground_truth_pts = gt_pts[i]         
                ground_truth_pts = ground_truth_pts*50.0+100
            
            # call show_all_keypoints
            self.show_landmarks(np.squeeze(image), predicted_key_pts, ground_truth_pts)
                
            plt.axis('off')

        plt.show()
    
# call it





#img = cv.imread("db-with-landmark/CollectionA/test/images/test_0000.png")
#crop_image(img, (10, 10, 100, 100))
