import os
import numpy as np
import torch.utils.data as data
import scipy.io as sio
import torch
import utils


# class HSTestData(data.Dataset):
#     def __init__(self, image_dir, use_3D=False):
#         test_data = sio.loadmat(image_dir)
#         self.use_3Dconv = use_3D
#         self.ms = np.array(test_data['ms'][...], dtype=np.float32)
#         self.lms = np.array(test_data['ms_bicubic'][...], dtype=np.float32)
#         self.gt = np.array(test_data['gt'][...], dtype=np.float32)
#         print(f"\n------HSTestData------")
#         print(f"HSTestData image_dir: {image_dir}")
#         print(f"HSTestData test_data shape:")
#         print(f"  ms: {test_data['ms'].shape}")
#         print(f"  ms_bicubic: {test_data['ms_bicubic'].shape}")
#         print(f"  gt: {test_data['gt'].shape}")
#         print(f"HSTestData ms shape: {self.ms.shape}")
#         print(f"HSTestData lms shape: {self.lms.shape}")
#         print(f"HSTestData gt shape: {self.gt.shape}")
#         print(f"HSTestData use_3Dconv: {self.use_3Dconv}\n")

#     def __getitem__(self, index):
#         ms = self.ms
#         lms = self.lms
#         gt = self.gt
#         if self.use_3Dconv:
#             gt = self.gt[index, :, :, :]
#             ms = self.ms[index, :, :, :]
#             lms = self.lms[index, :, :, :]
#             ms, lms, gt = ms[np.newaxis, :, :, :], lms[np.newaxis, :, :, :], gt[np.newaxis, :, :, :]
#             ms = torch.from_numpy(ms.copy()).permute(0, 3, 1, 2)
#             lms = torch.from_numpy(lms.copy()).permute(0, 3, 1, 2)
#             gt = torch.from_numpy(gt.copy()).permute(0, 3, 1, 2)
#         else:
#             ms = torch.from_numpy(ms.copy()).permute(2, 0, 1)
#             lms = torch.from_numpy(lms.copy()).permute(2, 0, 1)
#             gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
#         #ms = torch.from_numpy(ms.transpose((2, 0, 1)))
#         #lms = torch.from_numpy(lms.transpose((2, 0, 1)))
#         #gt = torch.from_numpy(gt.transpose((2, 0, 1)))
#         return ms, lms, gt

#     def __len__(self):
#         # return self.gt.shape[0]
#         return 1


def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])


class HSTestData(data.Dataset):
    def __init__(self, image_dir, augment=None, use_3D=False):
        self.image_files = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if is_mat_file(x)]
        self.augment = augment
        self.use_3Dconv = use_3D
        if self.augment:
            self.factor = 8
        else:
            self.factor = 1
        print(f"\n------HSTestData------")
        print(f"HSTestData image_files: {self.image_files[0]}")
        print(f"HSTestData factor: {self.factor}")
        print(f"HSTestData use_3Dconv: {self.use_3Dconv}")
        
        # Print shape of one of the files
        sample_data = sio.loadmat(self.image_files[0])
        print(f"Shape of 'ms' in first file: {sample_data['ms'].shape}")
        print(f"Shape of 'ms_bicubic' in first file: {sample_data['ms_bicubic'].shape}")
        print(f"Shape of 'gt' in first file: {sample_data['gt'].shape}")

    def __getitem__(self, index):
        file_index = index
        aug_num = 0
        if self.augment:
            file_index = index // self.factor   #
            aug_num = int(index % self.factor)  # 0-7
        load_dir = self.image_files[file_index]
        data = sio.loadmat(load_dir)
        ms = np.array(data['ms'][...], dtype=np.float32)
        lms = np.array(data['ms_bicubic'][...], dtype=np.float32)
        gt = np.array(data['gt'][...], dtype=np.float32)
        ms, lms, gt = utils.data_augmentation(ms, mode=aug_num), utils.data_augmentation(lms, mode=aug_num), utils.data_augmentation(gt, mode=aug_num)
        if self.use_3Dconv:
            ms, lms, gt = ms[np.newaxis, :, :, :], lms[np.newaxis, :, :, :], gt[np.newaxis, :, :, :]
            ms = torch.from_numpy(ms.copy()).permute(0, 3, 1, 2)
            lms = torch.from_numpy(lms.copy()).permute(0, 3, 1, 2)
            gt = torch.from_numpy(gt.copy()).permute(0, 3, 1, 2)
        else:
            ms = torch.from_numpy(ms.copy()).permute(2, 0, 1)
            lms = torch.from_numpy(lms.copy()).permute(2, 0, 1)
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return ms, lms, gt

    def __len__(self):
        return len(self.image_files)*self.factor
