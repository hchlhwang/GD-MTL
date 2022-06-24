import os
import torch
from torch.utils.data.dataset import Dataset
import numpy

class NYUV2(Dataset):
    def __init__(self, root, train=True, augmentation=False):
        self.train = train
        self.augmentation = augmentation

        if self.train:
            self.dataPath = os.path.expanduser(os.path.join(root, 'train'))
        else:
            self.dataPath = os.path.expanduser(os.path.join(root, 'val'))

        # Length of data
        self.dataLength = len(os.listdir(os.path.join(self.dataPath, 'image')))
        # TODO: Check which method is faster
        # self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))

    def __getitem__(self, index):
        # Load pre-preprocessed data
        try:
            # (288, 284, 3) -> (3, 288, 284)
            image = torch.from_numpy(np.moveaxis(np.load(os.path.join(self.dataPath, '/image/{:d}.npy'.format(index))), -1, 0))
            # (288, 284) -> (1, 288, 284)
            # TODO: different with original
            semantic = torch.from_numpy(np.expand_dims(np.load(os.path.join(self.dataPath, '/label/{:d}.npy'.format(index))), axis=0))
            # (288, 284, 1) -> (1, 288, 284)
            depth = torch.from_numpy(np.moveaxis(np.load(os.path.join(self.dataPath, '/depth/{:d}.npy'.format(index))), -1, 0))
            # (288, 284, 3) -> (3, 288, 284)
            normal = torch.from_numpy(np.moveaxis(np.load(os.path.join(self.dataPath, '/normal/{:d}.npy'.format(index))), -1, 0))


            return image.float(), semantic.float(), depth.float(), normal.float()

        except:
            return None

    def __len__(self):
        return self.dataLength
