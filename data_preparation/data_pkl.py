import os
import pandas as pd
import numpy as np
import sys
import pickle
from tqdm import tqdm

def min_max_scaling_along_axis(arr, axis):
    min_vals = np.min(arr, axis=axis, keepdims=True)
    max_vals = np.max(arr, axis=axis, keepdims=True)
    scaled_data = (arr - min_vals) / (max_vals - min_vals + 1e-7)
    return np.round(scaled_data, decimals=4)

def format_from_df(df):
    # 68 x,y pairs face points
    face = df.iloc[:, 299:435].values
    face = face.reshape(-1,2,68)
    scaled_face = min_max_scaling_along_axis(arr = face, axis = 2)
    scaled_face = scaled_face.transpose(0,2,1)

    # 17 AU intensity, since the scale is 0-5, rescale to 0-1
    aus = df.iloc[:, 679:696].values / 5

    # head pose coordinate
    head = min_max_scaling_along_axis(arr = df.iloc[:, 293:296].values, axis = 0)
    direction = min_max_scaling_along_axis(arr = df.iloc[:, 296:299].values, axis = 1)
    head = np.concatenate((head, direction), axis=1)
    return {'face':scaled_face, 'aus': aus, 'head': head}

'''
list[
    (exp{face:xx, 'aus':xx, 'head':xx}, 
    nov{face:xx, 'aus':xx, 'head':xx}),
    ...
]
'''

def main():
    data_folder = './data/openface'
    csv_list = './data/sample_list.csv'
    sample_list = pd.read_csv(os.path.join(csv_list)).iloc[:,1].values
    # print(self.sample_list)

    # debug
    pd.set_option('display.max_columns', None)
    lens_list = []

    data = []
    for item in tqdm(sample_list, desc='Formating openface data', leave=True):
        tmp_exp = pd.read_csv(os.path.join(data_folder, item+'_expert.csv'))
        tmp_nov = pd.read_csv(os.path.join(data_folder, item+'_novice.csv'))

        '''
        debugging section
        '''
        # print(tmp_exp.head())
        # 0-4:frame,face_id,timestamp,confidence,success (5 values)
        # print(tmp_exp.iloc[:, 0:5].head())
        # 5-12: gaze data (8 values)
        # print(tmp_exp.iloc[:, 5:13].head())
        #  13-124: eye landmarks 2d(112 values)
        # print(tmp_exp.iloc[:, 13:125].head())            
        #  125-292: eye landmarks 3d(168 values)
        # print(tmp_exp.iloc[:, 125:293].head())
        #  293-298: pose (6 values)
        # print(tmp_exp.iloc[:, 293:299].head())
        #  299- 434: face 2d (136 values)
        # print(tmp_exp.iloc[:, 299:435].head())
        #  435- 638: face 3d (204 values)
        # print(tmp_exp.iloc[:, 435:639].head())
        # 639 - 678: shape parameters (6+ 34 values)
        # print(tmp_exp.iloc[:, 639:679].head())
        # 679 - 713: action units (17 intensity + 18 presence)
        # print(tmp_exp.iloc[:, 679:714].head())
        
        # face 2d
        # tmp = tmp_exp.iloc[:, 299:435].head().values
        # print(tmp_exp.iloc[:, 299:435].head())
        # print(tmp.shape)
        # tmp = tmp.reshape(5,2,68)
        # print(tmp.shape)
        # print(tmp.transpose(0,2,1)) # got the (x , y) for every point



        # debugging section: data reformat
        # face = tmp_exp.iloc[:, 299:435].head().values
        # face = face.reshape(-1,2,68)
        # print(f'face in df: {tmp_exp.iloc[:, 299:435].head()}')
        # print(f"face: {face}")
        # video wise min-max scale normalization
        # scaled_face = min_max_scaling_along_axis(arr = face, axis = 2)
        # print((f"scaled_face: {scaled_face}"))
        # print(f'np.min {np.min(face,axis=2, keepdims=True)}')
        # face = face.reshape(-1,2,68).transpose(0,2,1)
        # todo after organized the data normalize all data to 0-1
        # (x, 68, 2) min-max normalization for each point
        # print(f"face.shape: {face.shape}")



        exp_ = format_from_df(tmp_exp)
        nov_ = format_from_df(tmp_nov)
        # print(f"exp_['face'].shape {exp_['face'].shape}")
        data.append((exp_,nov_))
        # print(f"{data}")

        # break
    with open('./data/test.pkl', 'wb') as file:
        pickle.dump(data, file)

    # with open('test.pkl', 'rb') as file:
    #     loaded_pickle = pickle.load(file)

    # print(loaded_pickle)
    # print(type(loaded_pickle[0][0]['face']))

if __name__ == '__main__':
    # main()
    with open('./data/test.pkl', 'rb') as file:
        loaded_pickle = pickle.load(file)
    # print(f'sys.getsizeof(loaded_pickle) {sys.getsizeof(loaded_pickle)}')
    # loaded_pick[sample index][exp or nov][face, head, aus]
    print(loaded_pickle[0][0]['face'].dtype)
    print(loaded_pickle[0][0]['head'].dtype)
    print(loaded_pickle[0][0]['aus'].dtype)