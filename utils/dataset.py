from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

import os
import numpy as np
from glob import glob
from PIL import Image
import torch
from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor
import random
import math


class Dataset_Traj(torch.utils.data.Dataset):

    def __init__(self, data_path, start, end, aug=False):
        self.aug = aug
        # read data
        all_data_list = []
        for i in range(start, end):
            file_name = data_path+'{}.npy'.format(i)
            if not os.path.exists(file_name):
                raise Exception("[!] {} not exists.".format(file_name))
            data = np.load(file_name)
            all_data_list.append(data)
            
        self.all_data = np.stack(all_data_list, axis=0)
        assert len(self.all_data.shape) == 3
        
        # get input, use feature_mask to select input feature
        feature_mask = [False, False, False, False, False, False, True, True, False, True, True, False, True, True, False]
        self.all_input = self.all_data[:,0,feature_mask]
        
        # normalize input
        self.in_mean, self.in_std = self.get_statistic_for_input(self.all_input)
        
        print('input data statistic:')
        print(self.in_mean)
        print(self.in_std)
        
        self.all_input = (self.all_input-self.in_mean)/self.in_std
        
        assert len(self.all_input.shape) == 2
        
        # get label
        self.all_labels =  self.all_data[:,:,2:4]
        
        # because we block the loss for ball out of the scene, we use a mask for the normalization of the label
        self.mask = self.get_mask(self.all_data)
        self.out_mask = self.mask[:,:,2:4]
        print(f'mask shape: {self.mask.shape}')
        # normalize label
        self.label_mean, self.label_std = self.get_statistic_for_label(self.all_labels, self.mask[:,:,2:4])
        print(f'label statistic: {self.label_mean},{self.label_std}')
        self.all_labels = (self.all_labels-self.label_mean)/self.label_std
       
        # To tensor
        self.all_input = torch.from_numpy(self.all_input).float()
        self.all_labels = torch.from_numpy(self.all_labels).float()
        
        # length and angle of the platforms, probably need to be changed for different dataset
        self.len_list = [90, 70, 80]
        self.theta_list = [-0.25*math.pi, 0, 0]
        
        
    def get_mask(self, data, x_min=0, x_max=600, y_min=0, y_max=600):    
        """ return a mask that indicate whether the prediction is invalid because the ball goes out of the scene
        """
        x_mask = np.logical_and(data[:,:,2] > x_min, data[:,:,2] < x_max)
        y_mask = np.logical_and(data[:,:,3] > y_min, data[:,:,3] < y_max)
        xy_mask = np.logical_and(x_mask, y_mask)
        mask = np.zeros(data.shape)
        assert len(data.shape) == 3
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if xy_mask[i,j]:
                    mask[i,j] = 1
        return mask
        
    def get_statistic_for_input(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return mean, std
    

    def get_statistic_for_label(self, data, mask):
        """ Get the statistics for the label with a mask.
        """
        data = np.ma.array(data=data, mask=np.logical_not(mask))
        mean = np.mean(data, axis=(0,1))
        std = np.std(data, axis=(0,1))
        return mean, std


    def __getitem__(self, index):
        net_input = self.all_input[index]
        net_output = self.all_labels[index]
        mask = self.out_mask[index]

        return {'net_input':net_input, 'net_output':net_output, 'valid_mask': mask}

    def __len__(self):
        return self.all_data.shape[0]

