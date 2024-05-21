import os
import numpy as np
from torch.utils.data import Dataset

# @LOAD USER ITEM INTERACTIONS
class ReviewData_test(Dataset):

    # DEFINE DATASET PATHS FOR LOAD AND BATCH SPLIT
    def __init__(self, root_path, mode, topic=False):

        if mode == 'Train':
            path = os.path.join(root_path, 'train/')
            print('loading train data')
            self.data = np.load(path + 'Train.npy', encoding='bytes')
            self.scores = np.load(path + 'Train_Score.npy')
        elif mode == 'Val':
            path = os.path.join(root_path, 'val/')
            print('loading val data')
            self.data = np.load(path + 'Val.npy', encoding='bytes')
            self.scores = np.load(path + 'Val_Score.npy')
        elif mode == 'Test':
            path = os.path.join(root_path, 'test/')
            print('loading test data')
            self.data = np.load(path + 'Test.npy', encoding='bytes')
            self.scores = np.load(path + 'Test_Score.npy')
        elif mode == 'All':
            print('loading all data')
            train_data = np.load(os.path.join(root_path, 'train/Train.npy'), encoding='bytes')
            train_scores = np.load(os.path.join(root_path, 'train/Train_Score.npy'))
            val_data = np.load(os.path.join(root_path, 'val/Val.npy'), encoding='bytes')
            val_scores = np.load(os.path.join(root_path, 'val/Val_Score.npy'))
            test_data = np.load(os.path.join(root_path, 'test/Test.npy'), encoding='bytes')
            test_scores = np.load(os.path.join(root_path, 'test/Test_Score.npy'))
            self.data = np.concatenate([train_data, val_data, test_data])
            self.scores = np.concatenate([train_scores, val_scores, test_scores])

        self.x = list(zip(self.data, self.scores))

    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]

    def __len__(self):
        return len(self.x)
    