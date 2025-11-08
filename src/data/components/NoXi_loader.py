import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle
class NoXiDataset(Dataset):
    ## mode 0 = both, 1 = expert only, 2 = novice only
    def __init__(self, test=False, stride = 5, trim_head = 5 * 25, trim_tail = 10 * 25, window_length = 10 *25):
        super().__init__()
        test = [0, 30, 60]
        data_folder = './data/openface.pkl'
        loaded_pickle = None
        with open(data_folder, 'rb') as file:
            loaded_pickle = pickle.load(file)
            if test:
                loaded_pickle = [loaded_pickle[i] for i in range(len(loaded_pickle)) if i in test]
            else:
                loaded_pickle = [loaded_pickle[i] for i in range(len(loaded_pickle)) if i not in test]
        print(f'pickle length {len(loaded_pickle)}')
        self.frame_lens = []
        for sample in loaded_pickle:
            self.frame_lens.append(len(sample[0]['face']))
        # print(f'frame_lens: {self.frame_lens}')

        # number of frames to skip for next sample
        self.stride = 5
        # number of frames to discard
        self.trim_head = 5 * 25
        self.trim_tail = 10 * 25
        self.window_length = 10 *25
        self.sample_lens = []
        self.total_len = 0
        for itm in self.frame_lens:
            tmp = (itm - self.trim_head - self.trim_tail - self.window_length) // self.stride
            self.sample_lens.append(tmp)
            self.total_len += tmp
        self.data = loaded_pickle
        # load label pickle
        data_folder = './data/label.pkl'
        with open(data_folder, 'rb') as file:
            loaded_pickle = pickle.load(file)
            if test:
                loaded_pickle = [loaded_pickle[i] for i in range(len(loaded_pickle)) if i in test]
            else:
                loaded_pickle = [loaded_pickle[i] for i in range(len(loaded_pickle)) if i not in test]
        
        self.label = loaded_pickle
        
        # print(f'data {self.data}')
        # print(f'label {self.label}')
        

        

    def idx_to_frame(self, idx):
        remain_idx = idx
        sample_idx = 0
        for itm in self.sample_lens:
            if itm > idx:
                break
            else:
                remain_idx -= itm
                sample_idx += 1
        frame_idx = remain_idx * self.stride + self.trim_head
        return (sample_idx, frame_idx)


    def __len__(self):
        return self.total_len * 2
    '''
        returns (x, y)
        x = (face, aus, head)
        face, aus, head = (windows size, feature dimision)
        y = scaler
    '''
    #lightning takes x,y = batch, so adjust accordingly
    def __getitem__(self, idx):
        remain_idx = idx
        # default person 0, i.e. expert
        target = 0
        # 0-total_len returns expert label and places expert in first slot of the tuple
        # total_len - 2* total_len returns novice label and places novice in first slot of the tuple
        if idx >= self.total_len:
            idx -= self.total_len
            target = 0
        sample_idx, frame_idx = self.idx_to_frame(remain_idx)

        face_ = self.data[sample_idx][target]['face'][sample_idx:sample_idx+self.window_length]
        aus_ = self.data[sample_idx][target]['aus'][sample_idx:sample_idx+self.window_length]
        head_ = self.data[sample_idx][target]['head'][sample_idx:sample_idx+self.window_length]

        # face (250, 68, 2) aus (250, 17) head (250, 6)
        # print(f'face {face_.shape} aus {aus_.shape} head {head_.shape}')
        x = (face_.reshape(self.window_length, -1), aus_, head_)

        y = self.label[sample_idx][target][frame_idx + self.window_length]
        

        return (x, y)





if __name__ == '__main__':
    sample_name = '001_2016-03-17_Paris'
    root = './processed_csv'
    window_length = 25 * 10
    print("fire")

    ds = NoXiDataset()
    sample = ds[0]
    # print(sample[0])
    # for f in sample[1]:
    #     print(f)
    print('Length is {length}'.format(length = len(ds)))
    print(sample)