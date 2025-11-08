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
    (exp sample1 label(np.array), 
    nov)
]

'''

def main():
    data_folder = './data/aria-noxi'
    csv_list = './data/sample_list.csv'
    sample_list = pd.read_csv(os.path.join(csv_list)).iloc[:,1].values
    # print(self.sample_list)

    # debug
    # pd.set_option('display.max_columns', None)
    lens_list = []


    data = []
    for item in tqdm(sample_list, desc='Formating engagement label', leave=True):
        df = pd.read_csv(os.path.join(data_folder, item, 'engagement.csv'), sep=';')

        # expert.engagement.challenge
        # expert.engagement.gold
        # select 'challenge gold' annotation first, if not exists 'gold' instead
        if 'expert.engagement.challenge' in df.columns:
            exp = df['expert.engagement.challenge']
            nov = df['novice.engagement.challenge']
        elif 'expert.engagement.gold' in df.columns:
            exp = df['expert.engagement.gold']
            nov = df['novice.engagement.gold']
        else:
            raise CustomError("No label selected")

        data.append((exp.values,nov.values))

        # break
    with open('./data/label.pkl', 'wb') as file:
        pickle.dump(data, file)

    with open('./data/label.pkl', 'rb') as file:
        loaded_pickle = pickle.load(file)

    print(loaded_pickle)
    print(type(loaded_pickle[0][0]))

if __name__ == '__main__':
    main()

    # with open('./data/test.pkl', 'rb') as file:
    #     loaded_pickle = pickle.load(file)
    # # print(f'sys.getsizeof(loaded_pickle) {sys.getsizeof(loaded_pickle)}')
    # # loaded_pick[sample index][exp or nov][face, head, aus]
    # print(loaded_pickle[0][0]['face'].dtype)
    # print(loaded_pickle[0][0]['head'].dtype)
    # print(loaded_pickle[0][0]['aus'].dtype)