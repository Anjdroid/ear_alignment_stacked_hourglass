from functools import reduce
import os

import cv2 as cv
import numpy as np
#import open3d
from scipy.io import loadmat
from torchvision import transforms, utils

from utils import Crop, Normalize, Rescale, ToTensor, Visualize, ToHeatmap, Equalize, Augment

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import torch
import json

BEST_LDS = [0,2,4,7,8,10,12,13,14,16,18,19,20,24,26,28,30,32,34,36,37,38,39,42,44,45,46,47,48,49,50,52,54]

class ReadData:
    def __init__(self, dir_2D, dir_PTC, dir_3DMM, htmps_path, transform=None, reduced=False, t_t = 'train', model='hourglass', num_aug=35, augment=False, num_lds=55):
        # MVP of directories
        self.dir_2D_images = dir_2D
        self.dir_2D_lds = htmps_path
        self.dir_PTC = dir_PTC
        self.dir_3DMM = dir_3DMM
        self.reduced = reduced
        self.num_lds = num_lds
        self.model_type = model
        self.augment = augment
        self.num_aug = num_aug

        # all the mothrfckn data
        self.images = {}
        self.ears_bbox = {}
        self.landmarks = {}
        self.landmarks_3D = {}
        self.ptc = {}
        self.mean_morphable_model = []
        self.morphable_models_pc = {}

        # transform operations
        self.transform = transform
        self.dims = (256, 256)
        #(256, 256)
        self.dataset = []
        self.vs = Visualize()

        self.t_t = t_t

        self.files = []

        self.htmps_path = htmps_path


    def init_transform_aug(self, num, path, num_lds):
        return transforms.Compose([
            Augment(num, path, num_lds)
        ])

    def init_transform_crop(self, crop_bbox):
        return transforms.Compose([
                    Crop(crop_bbox),  
                ])

    
    def init_transform_to_tensor(self, scale_dims, crop_bbox):
        return transforms.Compose([
                    Equalize(),
                    #Crop(crop_bbox),
                    #Rescale(scale_dims),
                    Normalize(),
                    ToTensor()
                    #ToHeatmap()    
                ])

    def init_transform_ht(self):
        return transforms.Compose([
                    #Normalize(),
                    #ToTensor(),
                    ToHeatmap()
                ])


    def generate_heatmaps(self, htmps, img_file):
        h = np.zeros(htmps[0].shape)
        #print(h.shape)

        for idx,item in enumerate(htmps):
            #print(idx)
            h = h + item
            matplotlib.image.imsave('db-with-landmark/CollectionA/cropped_data/htmps_train/' + img_file + '_' + str(idx) +'.jpg', item)
            #'db-with-landmark/CollectionA/augmented_data/' + self.t_t + '/heatmaps/' + img_file.split('.')[0] + '_' + str(idx) + '.jpg', item)
            #print(item)
            #cv.imshow('', item)
            #cv.imwrite('300W/Train/heatmaps/'+ img_file + '_' + str(idx) +'.jpg', item)
        #cv.imshow('',h)
        #cv.waitKey(0)

    def read_300w(self, path, path_ld):
        names = []
        dataset = []
        hts = []

        lala = False

        with open(path_ld) as f:
            lines = f.readlines()
            cnt = 0
            ct = 0
            for sample in lines:
                print(sample.split(' ')[0])
                
                samp = sample.split(' ')

                #print(samp[0])

                image = cv.cvtColor(cv.imread(path+samp[0]), cv.COLOR_BGR2RGB)
                #cv.imshow('', image)
                #cv.waitKey(0)
                names.append(samp[0].split('.')[0])

                #if sample.split(' ')[0] == 'image/helen/trainset/2965035072_1.jpg':
                #    lala = True
                lala = True
                
                if lala:
                    ## landmarks samp[1::]
                    landmarks = samp[1:-1]
                    #print(landmarks)

                    lds = np.zeros((55,2))
                    cnt = 0
                    for idx in range(0, len(landmarks), 2):
                        #print(len(landmarks))
                        #print(idx)
                        #print(landmarks[idx])
                        if cnt >= 55:
                            break
                        lds[cnt][0] = float(landmarks[idx])
                        lds[cnt][1] = float(landmarks[idx+1])
                        cnt += 1
                    #print(len(landmarks))
                    #landmarks = np.reshape(landmarks, (-1,2))
                    #print(landmarks.shape)
                    #print(sample['landmarks'].shape)

                    ld_x = lds[:,0]
                    ld_y = lds[:,1]
                    bbox = (ld_x.min(), ld_y.min(), ld_x.max(), ld_y.max())

                    tf2 = self.init_transform_to_tensor(self.dims, bbox)
                    sample = tf2({'image': image, 'landmarks': lds})

                    

                    

                    #tf = self.init_transform()
                    #ht = tf({'image': sample['image'], 'landmarks': sample['landmarks']})
                    #hts.append(ht)
                    #self.generate_heatmaps(ht['landmarks'], samp[0].split('/')[-1].split('.')[0])

                    #print(lds)
                

                    #tf1 = self.init_transform_to_tensor(self.dims, (0,0,0,0))
                    #sample = tf1({'image': image, 'landmarks': np.asarray([0])})
                    self.dataset.append({'image': sample['image'], 'landmarks': str(samp[0].split('.')[0])})
                    #cv.imshow('img', sample['image'].astype(np.uin))
                ct += 1

                if ct > 1:
                    break
        print(ct)


    def read_lds(self, ld_file, dir_2D_lds):
        with open(dir_2D_lds + ld_file) as f:
            rows = [rows.strip() for rows in f]

            # use the curly braces to find the start and end of the point data
            head = rows.index('{') + 1
            tail = rows.index('}')

            # select the point data split into coordinates
            raw_points = rows[head:tail]
            coords_set = [point.split() for point in raw_points]

            # convert entries from lists of strings to tuples of floats
            points = np.array([[float(point) for point in coords] for coords in coords_set])

            # use reduced point set
            if self.reduced:
                ct_ld = 0
                pts = np.zeros((len(BEST_LDS), 2))
                for idx,i in enumerate(points):
                    if idx in BEST_LDS:
                        pts[ct_ld] = i
                        ct_ld += 1
                points = pts.copy()
            else:
                ct_ld = 0
                pts = np.zeros((self.num_lds, 2))
                for idx,i in enumerate(points):
                    pts[ct_ld] = i
                    ct_ld += 1
                points = pts.copy()

        return points


    def calc_img_bbox(self, points):
        # crop images based on landmarks TODO:(=probably needs to be done)
        # define bbox by landmarks max/min
        ld_x = points[:,0]
        ld_y = points[:,1]
        bbox = (ld_x.min(), ld_y.min(), ld_x.max(), ld_y.max())

        # normalize landmarks
        #width = ld_x.max() - ld_x.min()
        #height = ld_y.max() - ld_y.min()
        return bbox


    
    def read_awe(self):
        #dir_groups = 'UERC Competition Package 2019/Dataset/Info Files/groups.txt'
        #with open(dir_groups,'r') as fi:
        #    impostors = [rows.strip() for rows in fi]
        self.dir_2D_images = '/'
        print("========== Reading 2D data ==========")
        img_folders = sorted(os.listdir('.'))
        #print(img_folders)
        cnt = 0
        #img_folders = img_folders
        for folder in img_folders:
            folder1 = self.dir_2D_images + folder
            #print("=flder=")
            #print(folder)

            #if folder == '1800':
            #    print(folder)
            #    break
            #for f in os.listdir('data/og_awe_ext_lds/train_joint/'+folder+'/'):
            #    os.remove('data/og_awe_ext_lds/train_joint/'+folder+'/'+f) 

            #for f in os.listdir('data/og_awe_unalign/'+folder+'/'):
            #    os.remove('data/og_awe_unalign/'+folder+'/'+f)

            imgs = sorted(os.listdir('.'))
            
            #js_f = open(folder1 +'/'+ imgs[-1],)
            # returns JSON object as
            # a dictionary
            #annotations = json.load(js_f)
            #annotations = annotations['data']
            #imgs = imgs

            for idx,file in enumerate(imgs):
                print(idx,file)
                img_file = folder1+ '/' + file
                #print(img_file)
                file = file.split('.')[0]
                #if annotations[file]['d'] == 'r':
                #print("RIGHT EAR")

                # points = self.read_lds(ld_file)
                # bbox = self.calc_img_bbox(points)

                #img_file = 'img_28.png'
                #ld_file = 'test_77.txt'
                    
                # read image data
                img = cv.imread(img_file)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                #plt.figure(frameon=False)
                #plt.imshow(img.astype(np.uint8))
                #plt.axis('off')
                #plt.savefig('data/og_awe_unalign/'+folder+'/' + file, bbox_inches='tight', pad_inches=0, transparent=True)
                #plt.show()
                #plt.pause(0.1)
                #plt.close()
                
                tf_crop = self.init_transform_crop([0,0,img.shape[1],img.shape[0]])
                cropped = tf_crop({'image': img, 'landmarks': None})

                img = cropped['image']
                og = np.copy(img)
                #lds = cropped['landmarks']

                if self.transform:
                    # first augment
                    if self.augment:
                        num_aug = self.num_aug
                        tf_aug = self.init_transform_aug(num_aug, self.htmps_path, self.num_lds)

                        augmented = tf_aug({'image': img, 'landmarks': torch.from_numpy(np.asarray([]))})
                        aug_imgs = augmented['image'] # LAST img is OG img
                        aug_lds = augmented['landmarks'] # LAST one are OG lds
                        #print(aug_lds)

                        tf = self.init_transform_to_tensor(self.dims, None)
                        for idx,item in enumerate(aug_imgs):
                            #    #print(idx, item)
                            sample = tf({'image': item, 'landmarks': None})
                            self.dataset.append({'image': sample['image'], 'landmarks': folder1+'/'+file.split('.')[0]+'_'+str(idx)+'.png'})
                    else:
                        tf = self.init_transform_to_tensor(self.dims, None)
                        sample = tf({'image':img, 'landmarks':[]})
                                                    
                        self.dataset.append({'image': sample['image'], 'landmarks': img_file, 'original':og})

            #if cnt == 0:
            ##    #    #    #if cnt == 10:
            #    break
            cnt += 1
            print("LEN OF DATASET")
            print(len(self.dataset))

        
        def read_collectionB(self):
            #dir_groups = 'UERC Competition Package 2019/Dataset/Info Files/groups.txt'
            #with open(dir_groups,'r') as fi:
            #    impostors = [rows.strip() for rows in fi]
            dir_2D_images = 'db-with-landmark/CollectionB/'
            print("========== Reading 2D data ==========")
            img_folders = sorted(os.listdir(dir_2D_images))
            #print(img_folders)
            cnt = 0
            img_folders = img_folders
            for folder in img_folders:
                folder1 = dir_2D_images + folder
                #print("=flder=")
                print(folder)

                #if folder == '1800':
                #    print(folder)
                #    break

                # all images for 1 subject
                imgs = sorted([file for file in os.listdir(folder1) if file.endswith('.png')])
                lds = sorted([file for file in os.listdir(folder1) if file.endswith('.pts')])

                print(len(imgs))
                print(len(lds))
                print(lds)
                
                for idx,file in enumerate(imgs):
                    print(idx,file)
                    img_file = folder1+ '/' + file
                    print(img_file)
                    file = file.split('.')[0]
                    ld_file = lds[idx]
                    print(ld_file)

                    points = self.read_lds(ld_file, folder1+'/')
                    #print(points)
                    bbox = points[:4]
                    #print(bbox)
                    bbox = self.calc_img_bbox(bbox)
                    print(bbox)

                    img = cv.imread(img_file)
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

                    tf_crop = self.init_transform_crop(bbox)
                    cropped = tf_crop({'image': img, 'landmarks': None})

                    img = cropped['image']
                    lds_ = cropped['landmarks']
                    cv.imshow('crop', img)
                    cv.waitKey(0)

                    if self.transform:
                        # first augment
                        if self.augment:
                            num_aug = self.num_aug
                            tf_aug = self.init_transform_aug(num_aug, self.htmps_path, self.num_lds)

                            augmented = tf_aug({'image': img, 'landmarks': None})
                            aug_imgs = augmented['image'] # LAST img is OG img
                            aug_lds = augmented['landmarks'] # LAST one are OG lds
                            #print(aug_lds)

                            tf = self.init_transform_to_tensor(self.dims, None)
                            for idx,item in enumerate(aug_imgs):
                                #    #print(idx, item)
                                sample = tf({'image': item, 'landmarks': None})
                                self.dataset.append({'image': sample['image'], 'landmarks': torch.from_numpy(np.asarray([]))})
                        else:
                            tf = self.init_transform_to_tensor(self.dims, None)
                            sample = tf({'image':img, 'landmarks':[]})
                                                        
                            self.dataset.append({'image': sample['image'], 'landmarks': folder1})

                    #if cnt == 10:
                    #    #    #    #if cnt == 10:
                    #    break
                    cnt += 1
        print("LEN OF DATASET")
        print(len(self.dataset))

        

    def read_images(self):
        print("========== Reading 2D data ==========")
        img_files = sorted(os.listdir(self.dir_2D_images))
        cnt = 0
        ffs = []
        for idx,file in enumerate(img_files):
            img_file = file
            file = file.split('.')[0]
            #file = "train_185"
            ld_file = file + ".txt"
            print(img_file, ld_file)

            # points = self.read_lds(ld_file)
            # bbox = self.calc_img_bbox(points)

            #img_file = 'test_77.png'
            #ld_file = 'test_77.txt'

            #print(self.dir_2D_images + img_file)
                
            # read image data
            img = cv.imread(self.dir_2D_images + img_file)
            #print(img.shape)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            lds = self.read_lds(ld_file)
            bbox = self.calc_img_bbox(lds)

            tf_crop = self.init_transform_crop(bbox)
            cropped = tf_crop({'image': img, 'landmarks': lds})

            img = cropped['image']
            lds = cropped['landmarks']

            if self.transform:
                # first augment
                if self.augment:
                    num_aug = self.num_aug
                    tf_aug = self.init_transform_aug(num_aug, self.htmps_path, self.num_lds)

                    augmented = tf_aug({'image': img, 'landmarks': lds})
                    aug_imgs = augmented['image'] # LAST img is OG img
                    aug_lds = augmented['landmarks'] # LAST one are OG lds
                    #print(aug_lds)

                    tf = self.init_transform_to_tensor(self.dims, bbox)
                    for idx,item in enumerate(aug_imgs):
                        #    #print(idx, item)
                        sample = tf({'image': item, 'landmarks': aug_lds[idx]})
                        self.dataset.append({'image': sample['image'], 'landmarks': sample['landmarks']})
                else:
                    tf = self.init_transform_to_tensor(self.dims, bbox)
                    sample = tf({'image':img, 'landmarks':lds})
                                                
                    self.dataset.append({'image': sample['image'], 'landmarks': sample['landmarks']})

            #if cnt == 0:
            #    #    #    #if cnt == 10:
            #    break
            cnt += 1
        print("LEN OF DATASET")
        print(len(self.dataset))


    def read_PTC(self):
        print("========== Reading PTCs ==========")
        dir_list = sorted(os.listdir(self.dir_PTC))

        for file in dir_list:
            filename = file.split(".")
            # print(filename)

            #if (filename[1] == file_ply):
            #ptc_file = open(self.dir_PTC + file, "r")

            # read with open3d
            cloud = open3d.io.read_point_cloud(self.dir_PTC + file)
            #print(cloud)
            #print(np.asarray(cloud.points))

            # print(np.asarray(cloud.points[920]))


            # shift + left clck to choose points
            """ http://www.open3d.org/docs/release/python_api/open3d.visualization.draw_geometries_with_vertex_selection.html#open3d.visualization.draw_geometries_with_vertex_selection"""
            # open3d.visualization.draw_geometries_with_vertex_selection([cloud], width=1400, height=900) # Visualize the point cloud

            #lines = [line.strip().split(" ") for line in ptc_file.readlines()]
            
            # skip PLY info in header
            #start_idx = 10
            #print(lines[start_idx])
            #print(lines[7121])
            
            # do not include vertex info
            #end_idx = 7121

            #ptc_data = lines[start_idx:end_idx]
            #ptc_data = np.asarray([[float(x[val]), float(x[val+1]), float(x[val+2])] for x in ptc_data for val in range(0,len(x),3)])
            #print(ptc_data)

            #print(ptc_data)

            self.ptc[filename[0]] = np.asarray(cloud.points)
            #print(self.ptc[filename[0]])
            #break

            # DEFINE 3D landmarks
            f = open('landmark_points.txt', 'r')
            pts = []
            lines = [int(line.strip().split("#")[1].split(" ")[0]) for line in f.readlines()]
            lds = self.ptc[filename[0]][lines]
            self.landmarks_3D[filename[0]] = lds
            #print(lds)
            #print(len(lines))
            break

        
            ## TODO
            # PLY file inclues also vertex info
            # property list uchar short vertex_indices
            # x, y, z til line 7122
            # this is done as: read_triangle_mesh


    def read_3DMM(self):
        print("========== Loading 3DMM ==========")
        morphable = loadmat(self.dir_3DMM)

        print("Header: ", morphable['__header__'])
        print("Version: ", morphable['__version__'])
        print("Globals: ", morphable['__globals__'])
        print("Coeff5: ", morphable['coeff5'])
        print("Latent5: ", morphable['latent5'])
        print("mu5: ", morphable['mu5'])

        coeff5 = morphable['coeff5']
        mu5 = morphable['mu5']
        latent5 = morphable['latent5']

        # triangle connectivity matrix
        print("sourceF1: ", morphable['sourceF1'])

        # mean ear shape
        self.mean_morphable_model = np.asarray(mu5).reshape(3, 7111)
        print("=== SIZE ===")
        print(len(self.mean_morphable_model))
        print(len(self.mean_morphable_model[0]))

        for i in range(len(coeff5[0])):
            #print(np.dot(coeff5[:,i],np.sqrt(latent5[i,1])))
            #print(np.sqrt(latent5[i,1]))
            #print(mu5)
            #print(len(mu5[0]))
            #print(coeff5[:,i])
            #print(len(coeff5[:,i]))
            #print(latent5[i,0])
            #print(np.sqrt(latent5[i,1])))

            #print(np.dot(coeff5[:,i],np.sqrt(latent5[i,0])))
            mm_pc = mu5 + np.transpose(np.dot(coeff5[:,i],np.sqrt(latent5[i,0])))
            #print(mm_pc)
            self.morphable_models_pc[i] = np.transpose(mm_pc.reshape(3, 7111))
            #self.morphable_models_pc[i] = mu5 + np.transpose(np.dot(coeff5[:,i] * np.sqrt(latent5[i,1]))).reshape(3, 7111)
            # reshape( mu5+(coeff5(:,i)*sqrt(latent5(i,1)))'*3,[3 7111])'
