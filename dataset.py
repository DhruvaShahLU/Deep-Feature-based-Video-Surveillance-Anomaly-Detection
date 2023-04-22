import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random


class Normal_Loader(Dataset):
    """
    A class for loading the normal dataset.
    """
    def __init__(self, is_train=1, path='./UCF-Crime', modality='TWO'):
        super(Normal_Loader, self).__init__()
        
        # is_train: 1 for training set, 0 for test set
        self.is_train = is_train
        
        # modality: 'RGB' for RGB data, 'FLOW' for optical flow data, and 'TWO' for concatenated RGB and optical flow data.
        self.modality = modality
        
        # path: path to the dataset directory
        self.path = path
        
        # If is_train is 1, read the train_normal.txt file to get the list of data files.
        # If is_train is 0, read the test_normalv2.txt file to get the list of data files, shuffle the list, and take all but the last 10 files.
        if self.is_train == 1:
            data_list = os.path.join(path, 'train_normal.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = os.path.join(path, 'test_normalv2.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
            random.shuffle(self.data_list)
            self.data_list = self.data_list[:-10]

    def __len__(self):
        """
        Returns the length of the data_list.
        """
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Returns the idx-th item from the data_list.
        """
        if self.is_train == 1:
            # Load the RGB and optical flow numpy arrays, concatenate them if modality is 'TWO', and return the appropriate data.
            rgb_npy = np.load(os.path.join(self.path+'/all_rgbs', self.data_list[idx][:-1]+'.npy'))
            flow_npy = np.load(os.path.join(self.path+'/all_flows', self.data_list[idx][:-1]+'.npy'))
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            if self.modality == 'RGB':
                return rgb_npy
            elif self.modality == 'FLOW':
                return flow_npy
            else:
                return concat_npy
        else:
            # Split the line in the data_list to get the name, number of frames, and ground truth label.
            # Load the RGB and optical flow numpy arrays, concatenate them if modality is 'TWO', and return the appropriate data with the frames and ground truth label.
            name, frames, gts = self.data_list[idx].split(' ')[0], int(self.data_list[idx].split(' ')[1]), int(self.data_list[idx].split(' ')[2][:-1])
            rgb_npy = np.load(os.path.join(self.path+'/all_rgbs', name + '.npy'))
            flow_npy = np.load(os.path.join(self.path+'/all_flows', name + '.npy'))
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            if self.modality == 'RGB':
                return rgb_npy, gts, frames
            elif self.modality == 'FLOW':
                return flow_npy, gts, frames
            else:
                return concat_npy, gts, frames


class Anomaly_Loader(Dataset):
    """
    A PyTorch dataset for loading anomaly detection dataset
    is_train = 1 <- train, 0 <- test
    """
    def __init__(self, is_train=1, path='./UCF-Crime', modality='TWO'):
        super(Anomaly_Loader, self).__init__()
        self.is_train = is_train
        self.modality = modality
        self.path = path
        
        # Load the data list based on whether the dataset is for training or testing
        if self.is_train == 1:
            data_list = os.path.join(path, 'train_anomaly.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = os.path.join(path, 'test_anomalyv2.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()

    def __len__(self):
        """
        Returns the number of samples in the dataset
        """
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset at the given index
        """
        if self.is_train == 1:
            # Load RGB and optical flow data for training samples and concatenate them
            rgb_npy = np.load(os.path.join(self.path+'/all_rgbs', self.data_list[idx][:-1]+'.npy'))
            flow_npy = np.load(os.path.join(self.path+'/all_flows', self.data_list[idx][:-1]+'.npy'))
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            
            # Return the data based on the selected modality
            if self.modality == 'RGB':
                return rgb_npy
            elif self.modality == 'FLOW':
                return flow_npy
            else:
                return concat_npy
        else:
            # Load RGB and optical flow data, ground truth labels, and number of frames for testing samples
            name, frames, gts = self.data_list[idx].split('|')[0], int(self.data_list[idx].split('|')[1]), self.data_list[idx].split('|')[2][1:-2].split(',')
            gts = [int(i) for i in gts]
            rgb_npy = np.load(os.path.join(self.path+'/all_rgbs', name + '.npy'))
            flow_npy = np.load(os.path.join(self.path+'/all_flows', name + '.npy'))
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
            
            # Return the data, ground truth labels, and number of frames based on the selected modality
            if self.modality == 'RGB':
                return rgb_npy, gts, frames
            elif self.modality == 'FLOW':
                return flow_npy, gts, frames
            else:
                return concat_npy, gts, frames

if __name__ == '__main__':
    # Create an instance of the dataset for testing
    loader2 = Normal_Loader(is_train=0)

    


