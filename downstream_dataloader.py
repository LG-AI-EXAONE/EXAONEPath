import os
import time
import random
import argparse
import sys
import h5py
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader

from downstream_utils import cleanup, init_distributed_mode, return_rank_world_size
from datasets.macenko_normalizer import TorchMacenkoNormalizer


class macenko_normalizer():
    def __init__(self, shared, target_path=None):
        if target_path is None:
            target_path = f'/data/{shared}/shared/js.yun/DINO_sagemaker/dino_pl/macenko_target/target_TCGA-55-A48X_coords_[19440  9824]_[4096 4096].png'
        self.target = Image.open(target_path)
        self.transform_before_macenko = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x*255)
        ])
        self.normalizer = TorchMacenkoNormalizer()
        self.normalizer.fit(self.transform_before_macenko(self.target))

    def __call__(self, image):
        t_to_transform = self.transform_before_macenko(image)
        norm, _, _ = self.normalizer.normalize(I=t_to_transform, stains=False, form='chw', dtype='float')
        return norm

def make_transforms(basic_transforms, macenko, resize, crop, hflip, vflip, rotation, jitter, normalize):
    # Initialize the list of transformations
    train_transforms = basic_transforms.copy()
    test_transforms = basic_transforms.copy()

    if resize:
        train_transforms.append(transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC))
        test_transforms.append(transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC))
        if crop:    # crop always used with resize 
            train_transforms.append(transforms.CenterCrop(crop))
            test_transforms.append(transforms.CenterCrop(crop))

    # Horizontal flip (only applied to training)
    if hflip:
        train_transforms.append(transforms.RandomHorizontalFlip(p=0.5))

    # Vertical flip (only applied to training)
    if vflip:
        train_transforms.append(transforms.RandomVerticalFlip(p=0.5))

    # Rotation (only applied to training)
    if rotation:
        def random_rotate(img):
            angles = [0, 90, 180, 270]
            angle = random.choice(angles)
            return F.rotate(img, angle)
        # train_transforms.append(transforms.RandomApply([transforms.Lambda(random_rotate)], p=0.5))
        train_transforms.append(transforms.Lambda(random_rotate))

    # Color jitter (only applied to training)
    if jitter:
        train_transforms.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1))

    # Convert to tensor - macenko는 이미 tesnor로 들어옴
    if not macenko:
        train_transforms.append(transforms.ToTensor())
        test_transforms.append(transforms.ToTensor())

    # Normalize
    if normalize is None:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms.append(normalize)
    test_transforms.append(normalize)

    # Combine the transforms into a Compose object
    train_transform = transforms.Compose(train_transforms)
    test_transform = transforms.Compose(test_transforms)

    return train_transform, test_transform


#############################################################################################
# Custom Dataset class for loading PCAM_Dataset data
class PCAM_Dataset(IterableDataset):
    def __init__(self, x_path, y_path, macenko=False, transform=None, shuffle=False, return_idx=False, trim=False):
        self.x_path = x_path
        self.y_path = y_path
        self.macenko = macenko
        self.transform = transform
        self.shuffle = shuffle
        self.return_idx = return_idx
        self.trim = trim
        self.length = len(h5py.File(self.y_path, 'r')['y'])

        # Load data into memory
        with h5py.File(self.x_path, 'r') as f:
            self.images = f['x'][:]
        with h5py.File(self.y_path, 'r') as f:
            self.labels = f['y'][:]
            self.labels = torch.tensor(self.labels, dtype=torch.long)  # Convert labels to LongTensor
        self.data_indices = np.arange(self.length)

        # Shuffle data
        if self.shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        # Save the current random state
        state = np.random.get_state()

        # Set a fixed random seed for reproducibility
        np.random.seed(42)
        np.random.shuffle(self.data_indices)
        self.images = self.images[self.data_indices]
        self.labels = self.labels[self.data_indices]

        # Restore the previous random state
        np.random.set_state(state)

    def __len__(self):
        return self.length

    def worker_init_fn(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        num_workers = worker_info.num_workers
        self.worker_id = worker_id

        rank, world_size = return_rank_world_size()
        self.global_worker_id = rank * num_workers + worker_id

        # Trim the dataframe to make its length a multiple of world_size
        if self.trim:
            total_rows = len(self.images)
            rows_to_trim = total_rows % world_size
            if rows_to_trim != 0:
                self.images = self.images[:-rows_to_trim]
                self.labels = self.labels[:-rows_to_trim]
                self.data_indices = np.arange(len(self.images))

        # rank_sharding
        self.images = self.images[round(len(self) * rank / world_size):round(len(self) * (rank + 1) / world_size)]
        self.labels = self.labels[round(len(self) * rank / world_size):round(len(self) * (rank + 1) / world_size)]
        self.data_indices = self.data_indices[round(len(self) * rank / world_size):round(len(self) * (rank + 1) / world_size)]

        # worker sharding
        dataset.images = self.images[worker_id::num_workers]
        dataset.labels = self.labels[worker_id::num_workers]
        dataset.data_indices = self.data_indices[worker_id::num_workers]
        print(f'global worker: {self.global_worker_id}, len: {len(self.images)}')

    def __iter__(self):
        if self.shuffle:
            # Use the worker's unique seed to shuffle the dataframe
            seed = torch.initial_seed() % (2**32 - 1)
            # print(f'{self.global_worker_id} {seed}')
            random.seed(seed + self.worker_id)
            indices = np.arange(len(self.images))
            np.random.shuffle(indices)
            self.images = self.images[indices]
            self.labels = self.labels[indices]
            self.data_indices = self.data_indices[indices]

        for idx in range(len(self.images)):
            image = self.images[idx]                # tensor (chw)
            label = self.labels[idx].squeeze()

            if self.macenko:
                try:
                    image = self.macenko(image)
                except Exception as e:
                    if 'linalg.eigh:' in str(e) or "kthvalue()" in str(e):
                        pass
                    else:
                        print(e)
                    continue
                if torch.any(torch.isnan(image)):
                    continue

            if self.transform:
                image = self.transform(image)
            if self.return_idx:
                data_idx =  self.data_indices[idx]
                yield image, label, data_idx
            else:
                yield image, label
                # yield np.array(image), label

# PCAM dataloader creation function
def PCAM_dataloader(
        data_path, 
        batch_size=128, 
        num_workers=4, 
        shuffle=False, 
        drop_last=True, 
        macenko=False, 
        resize=None, 
        crop=None, 
        hflip=False, 
        vflip=False, 
        rotation=False, 
        jitter=False, 
        return_idx=False,
        trim=False,
        normalize=None):
    
    # File paths
    train_x_path = os.path.join(data_path, 'camelyonpatch_level_2_split_train_x.h5')    # (262144, 96, 96, 3)
    train_y_path = os.path.join(data_path, 'camelyonpatch_level_2_split_train_y.h5')
    val_x_path = os.path.join(data_path, 'camelyonpatch_level_2_split_valid_x.h5')
    val_y_path = os.path.join(data_path, 'camelyonpatch_level_2_split_valid_y.h5')
    test_x_path = os.path.join(data_path, 'camelyonpatch_level_2_split_test_x.h5')
    test_y_path = os.path.join(data_path, 'camelyonpatch_level_2_split_test_y.h5')

    basic_transforms = []
    if not macenko:
        basic_transforms.append(transforms.ToPILImage())    # ndarray -> PIL
    train_transform, test_transform = make_transforms(basic_transforms, macenko=macenko, resize=resize, crop=crop, hflip=hflip, vflip=vflip, rotation=rotation, jitter=jitter, normalize=normalize)

    # Datasets
    train_dataset = PCAM_Dataset(train_x_path, train_y_path, macenko=macenko, transform=train_transform, shuffle=shuffle, return_idx=return_idx, trim=trim)
    val_dataset = PCAM_Dataset(val_x_path, val_y_path, macenko=macenko, transform=test_transform, return_idx=return_idx, trim=trim)
    test_dataset = PCAM_Dataset(test_x_path, test_y_path, macenko=macenko, transform=test_transform, return_idx=return_idx, trim=trim)
    print(f'PCAM train/val/test size: {len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}')
    print(f'PCAM total size: {len(train_dataset) + len(val_dataset) + len(test_dataset)}')

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, \
                            worker_init_fn=train_dataset.worker_init_fn, pin_memory=True, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, \
                            worker_init_fn=val_dataset.worker_init_fn, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, \
                             worker_init_fn=test_dataset.worker_init_fn, pin_memory=True, drop_last=False)
        
    return train_loader, val_loader, test_loader


#############################################################################################
# MHIST dataset
class MHIST_Dataset(IterableDataset):
    def __init__(self, dataframe, images_dir, macenko=False, transform=None, shuffle=False, return_idx=False, trim=False):
        self.dataframe = dataframe      # csv 이미 shuffle 시켜놨음
        self.images_dir = images_dir
        self.macenko = macenko
        self.transform = transform
        self.shuffle = shuffle 
        self.return_idx = return_idx
        self.trim = trim

    def __len__(self):
        return len(self.dataframe)

    def worker_init_fn(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers
        self.worker_id = worker_id

        rank, world_size = return_rank_world_size()
        self.global_worker_id = rank*num_workers + worker_id

        # Trim the dataframe to make its length a multiple of world_size
        if self.trim:
            total_rows = len(self.dataframe)
            rows_to_trim = total_rows % world_size
            if rows_to_trim != 0:
                self.dataframe = self.dataframe[:-rows_to_trim].reset_index(drop=True)

        # rank sharding
        self.dataframe = self.dataframe[rank::world_size]
        # worker sharding
        self.dataframe = self.dataframe[worker_id::num_workers]
        # print(f'global worker: {rank*num_workers+worker_id}, len: {len(self.dataframe)}')

    def __iter__(self):
        if self.shuffle:
            # Use the worker's unique seed to shuffle the dataframe
            seed = torch.initial_seed() % (2**32 - 1)
            random.seed(seed + self.worker_id)
            self.dataframe = self.dataframe.sample(frac=1, random_state=seed + self.worker_id)
        
        for idx in range(len(self.dataframe)):
            img_name = self.dataframe.iloc[idx]['Image Name']
            img_path = os.path.join(self.images_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            label = self.dataframe.iloc[idx]['Majority Vote Label']
            label = 1 if label == 'SSA' else 0  # Assuming binary classification between 'SSA' and 'HP'
            
            if self.macenko:
                try:
                    image = self.macenko(image)
                except Exception as e:
                    if 'linalg.eigh:' in str(e) or "kthvalue()" in str(e):
                        pass
                    else:
                        print(e)
                    continue
                if torch.any(torch.isnan(image)):
                    continue
            if self.transform:
                image = self.transform(image)
            if self.return_idx:
                yield image, label, self.dataframe.index[idx]
            else:
                yield image, label    
                # yield np.array(image), label

def MHIST_dataloader(
        data_path, 
        batch_size=128, 
        num_workers=4, 
        shuffle=False, 
        drop_last=True, 
        macenko=False, 
        resize=None, 
        crop=None, 
        hflip=False, 
        vflip=False, 
        rotation=False, 
        jitter=False, 
        return_idx=False,
        trim=False,
        normalize=None
        ):
    
    csv_path = os.path.join(data_path, 'annotations_js.csv')
    images_dir = os.path.join(data_path, 'images')

    # 데이터프레임 로드
    df = pd.read_csv(csv_path)

    # 데이터셋 분리
    train_df = df[df['Partition'] == 'train'].reset_index(drop=True)
    val_df = df[df['Partition'] == 'val'].reset_index(drop=True)
    test_df = df[df['Partition'] == 'test'].reset_index(drop=True)

    basic_transforms = []
    train_transform, test_transform = make_transforms(basic_transforms, macenko=macenko, resize=resize, crop=crop, hflip=hflip, vflip=vflip, rotation=rotation, jitter=jitter, normalize=normalize)

    # 데이터셋 생성
    train_dataset = MHIST_Dataset(train_df, images_dir, macenko=macenko, transform=train_transform, shuffle=shuffle, return_idx=return_idx, trim=trim)
    val_dataset = MHIST_Dataset(val_df, images_dir, macenko=macenko, transform=test_transform, return_idx=return_idx, trim=trim)
    test_dataset = MHIST_Dataset(test_df, images_dir, macenko=macenko, transform=test_transform, return_idx=return_idx, trim=trim)
    print(f'MHIST tran/val/test size: {len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}')
    print(f'MHIST total size: {len(train_dataset) + len(val_dataset) + len(test_dataset)}')

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, \
                            worker_init_fn=train_dataset.worker_init_fn, pin_memory=True, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, \
                            worker_init_fn=val_dataset.worker_init_fn, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, \
                             worker_init_fn=test_dataset.worker_init_fn, pin_memory=True, drop_last=False)
        
    return train_loader, val_loader, test_loader


#############################################################################################
# CRC100k
class CRC100k_Dataset(IterableDataset):
    def __init__(self, dataframe, images_dir, macenko=False, transform=None, shuffle=False, return_idx=False, trim=False):
        self.dataframe = dataframe      # csv shuffle 되어있음
        self.images_dir = images_dir
        self.macenko = macenko
        self.transform = transform
        self.shuffle = shuffle
        self.return_idx = return_idx
        self.trim = trim
        self.label_dict = self.label_dict = {'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'MUC': 4, 'MUS': 5, 'NORM': 6, 'STR': 7, 'TUM': 8}

    def __len__(self):
        return len(self.dataframe)

    def worker_init_fn(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers
        self.worker_id = worker_id

        rank, world_size = return_rank_world_size()
        self.global_worker_id = rank*num_workers + worker_id
     
        # Trim the dataframe to make its length a multiple of world_size
        if self.trim:
            total_rows = len(self.dataframe)
            rows_to_trim = total_rows % world_size
            if rows_to_trim != 0:
                self.dataframe = self.dataframe[:-rows_to_trim].reset_index(drop=True)
       
        # rank sharding
        self.dataframe = self.dataframe[rank::world_size]
        # worker sharding
        self.dataframe = self.dataframe[worker_id::num_workers]

    def __iter__(self):
        if self.shuffle:
            # Use the worker's unique seed to shuffle the dataframe
            seed = torch.initial_seed() % (2**32 - 1)
            random.seed(seed + self.worker_id)
            self.dataframe = self.dataframe.sample(frac=1, random_state=seed + self.worker_id)
        
        for idx in range(len(self.dataframe)):
            img_name = self.dataframe.iloc[idx]['image_path']
            img_path = os.path.join(self.images_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            label = self.dataframe.iloc[idx]['label']
            # Convert label to numerical value using the dictionary
            label = self.label_dict[label]
            
            if self.macenko:
                try:
                    image = self.macenko(image)
                except Exception as e:
                    if 'linalg.eigh:' in str(e) or "kthvalue()" in str(e):
                        pass
                    else:
                        print(e)
                    continue
                if torch.any(torch.isnan(image)):
                    continue
            if self.transform:
                image = self.transform(image)
            if self.return_idx:
                yield image, label, self.dataframe.index[idx]
            else:
                yield image, label    
                # yield np.array(image), label

def CRC100k_dataloader(
        data_path, 
        batch_size=128, 
        num_workers=4, 
        shuffle=False, 
        drop_last=True, 
        macenko=False, 
        resize=None, 
        crop=None, 
        hflip=False, 
        vflip=False, 
        rotation=False, 
        jitter=False,
        return_idx=False,
        trim=False,
        normalize=None
        ):

    csv_path = os.path.join(data_path, 'data_split_js.csv')
    images_dir = data_path

    # 데이터프레임 로드
    df = pd.read_csv(csv_path)

    # 데이터셋 분리
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    basic_transforms = []
    train_transform, test_transform = make_transforms(basic_transforms, macenko=macenko, resize=resize, crop=crop, hflip=hflip, vflip=vflip, rotation=rotation, jitter=jitter, normalize=normalize)

    # 데이터셋 생성
    train_dataset = CRC100k_Dataset(train_df, images_dir, macenko=macenko, transform=train_transform, shuffle=shuffle, return_idx=return_idx, trim=trim)
    val_dataset = CRC100k_Dataset(val_df, images_dir, macenko=macenko, transform=test_transform, return_idx=return_idx, trim=trim)
    test_dataset = CRC100k_Dataset(test_df, images_dir, macenko=macenko, transform=test_transform, return_idx=return_idx, trim=trim)
    print(f'CRC100k tran/val/test size: {len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}')
    print(f'CRC100k total size: {len(train_dataset) + len(val_dataset) + len(test_dataset)}')

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, \
                            worker_init_fn=train_dataset.worker_init_fn, pin_memory=True, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, \
                            worker_init_fn=val_dataset.worker_init_fn, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, \
                             worker_init_fn=test_dataset.worker_init_fn, pin_memory=True, drop_last=False)
    
    return train_loader, val_loader, test_loader


#############################################################################################
class TIL_Det_Dataset(IterableDataset):
    def __init__(self, dataframe, images_dir, macenko=False, transform=None, shuffle=False, return_idx=False, trim=False):
        self.dataframe = dataframe          # csv에 이미 shuffle 시켜놨음
        self.images_dir = images_dir
        self.macenko = macenko
        self.transform = transform
        self.shuffle = shuffle
        self.return_idx = return_idx
        self.trim = trim
        self.label_dict = {'til-negative': 0, 'til-positive': 1}

    def __len__(self):
        return len(self.dataframe)

    def worker_init_fn(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers
        self.worker_id = worker_id

        rank, world_size = return_rank_world_size()
        self.global_worker_id = rank*num_workers + worker_id

        # Trim the dataframe to make its length a multiple of world_size
        if self.trim:
            total_rows = len(self.dataframe)
            rows_to_trim = total_rows % world_size
            if rows_to_trim != 0:
                self.dataframe = self.dataframe[:-rows_to_trim].reset_index(drop=True)

        # rank sharding
        self.dataframe = self.dataframe[rank::world_size]
        # worker sharding
        self.dataframe = self.dataframe[worker_id::num_workers]
        print(f'global worker: {rank*num_workers+worker_id}, len: {len(self.dataframe)}')

    def __iter__(self):
        if self.shuffle:
            # Use the worker's unique seed to shuffle the dataframe
            seed = torch.initial_seed() % (2**32 - 1)
            random.seed(seed + self.worker_id)
            self.dataframe = self.dataframe.sample(frac=1, random_state=seed + self.worker_id)

        for idx in range(len(self.dataframe)):
            img_name = self.dataframe.iloc[idx]['path']
            img_path = os.path.join(self.images_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            label = self.dataframe.iloc[idx]['label']
            # Convert label to numerical value using the dictionary
            label = self.label_dict[label]
            
            if self.macenko:
                try:
                    image = self.macenko(image)
                except Exception as e:
                    if 'linalg.eigh:' in str(e) or "kthvalue()" in str(e):
                        pass
                    else:
                        print(e)
                    continue
                if torch.any(torch.isnan(image)):
                    continue
            if self.transform:
                image = self.transform(image)
            if self.return_idx:
                yield image, label, self.dataframe.index[idx]
            else:
                yield image, label    
                # yield np.array(image), label

def TIL_Det_dataloader(
        data_path, 
        batch_size=128, 
        num_workers=4, 
        shuffle=False, 
        drop_last=True, 
        macenko=False, 
        resize=None, 
        crop=None, 
        hflip=False, 
        vflip=False, 
        rotation=False, 
        jitter=False,
        return_idx=False,
        trim=False,
        normalize=None
        ):
    
    csv_path = os.path.join(data_path, 'data_split_js.csv')
    images_dir = data_path

    # 데이터프레임 로드
    df = pd.read_csv(csv_path)

    # 데이터셋 분리
    train_df = df[df['partition'] == 'train'].reset_index(drop=True)
    val_df = df[df['partition'] == 'val'].reset_index(drop=True)
    test_df = df[df['partition'] == 'test'].reset_index(drop=True)

    basic_transforms = []
    train_transform, test_transform = make_transforms(basic_transforms, macenko=macenko, resize=resize, crop=crop, hflip=hflip, vflip=vflip, rotation=rotation, jitter=jitter, normalize=normalize)

    # 데이터셋 생성
    train_dataset = TIL_Det_Dataset(train_df, images_dir, macenko=macenko, transform=train_transform, shuffle=shuffle, return_idx=return_idx, trim=trim)
    val_dataset = TIL_Det_Dataset(val_df, images_dir, macenko=macenko, transform=test_transform, return_idx=return_idx, trim=trim)
    test_dataset = TIL_Det_Dataset(test_df, images_dir, macenko=macenko, transform=test_transform, return_idx=return_idx, trim=trim)
    print(f'TIL_Det tran/val/test size: {len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}')
    print(f'TIL_Det total size: {len(train_dataset) + len(val_dataset) + len(test_dataset)}')

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, \
                            worker_init_fn=train_dataset.worker_init_fn, pin_memory=True, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, \
                            worker_init_fn=val_dataset.worker_init_fn, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, \
                             worker_init_fn=test_dataset.worker_init_fn, pin_memory=True, drop_last=False)
    
    return train_loader, val_loader, test_loader


#############################################################################################
class MSI_CRC_Dataset(IterableDataset):
    def __init__(self, dataframe, images_dir, macenko=False, transform=None, shuffle=False, return_idx=False, trim=False):
        self.dataframe = dataframe
        self.images_dir = images_dir
        self.macenko = macenko
        self.transform = transform
        self.shuffle = shuffle
        self.return_idx = return_idx
        self.trim = trim

    def __len__(self):
        return len(self.dataframe)

    def worker_init_fn(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers
        self.worker_id = worker_id

        rank, world_size = return_rank_world_size()
        self.global_worker_id = rank*num_workers + worker_id
        # print(f'{self.global_worker_id}, df {self.dataframe.iloc[0]}')

        # Trim the dataframe to make its length a multiple of world_size
        if self.trim:
            total_rows = len(self.dataframe)
            rows_to_trim = total_rows % world_size
            if rows_to_trim != 0:
                self.dataframe = self.dataframe[:-rows_to_trim].reset_index(drop=True)

        # rank sharding
        self.dataframe = self.dataframe[rank::world_size]
        # worker sharding
        self.dataframe = self.dataframe[worker_id::num_workers]
        print(f'global worker: {rank*num_workers+worker_id}, len: {len(self.dataframe)}')

    def __iter__(self):
        if self.shuffle:
            # Use the worker's unique seed to shuffle the dataframe
            seed = torch.initial_seed() % (2**32 - 1)
            random.seed(seed + self.worker_id)
            self.dataframe = self.dataframe.sample(frac=1, random_state=seed + self.worker_id)
        
        for idx in range(len(self.dataframe)):
            img_path = os.path.join(self.images_dir, self.dataframe.iloc[idx]['image_path'])
            image = Image.open(img_path).convert('RGB')
            label = self.dataframe.iloc[idx]['label']
            label = 1 if label == 'MSIMUT' else 0
        
            if self.macenko:
                try:
                    image = self.macenko(image)
                except Exception as e:
                    if 'linalg.eigh:' in str(e) or "kthvalue()" in str(e):
                        pass
                    else:
                        print(e)
                    continue
                if torch.any(torch.isnan(image)):
                    continue
            if self.transform:
                image = self.transform(image)
            if self.return_idx:
                yield image, label, self.dataframe.index[idx]
            else:
                yield image, label    
                # yield np.array(image), label
    
def MSI_CRC_dataloader(
        data_path, 
        batch_size=128, 
        num_workers=4, 
        shuffle=False, 
        drop_last=True, 
        macenko=False, 
        resize=None, 
        crop=None, 
        hflip=False, 
        vflip=False, 
        rotation=False, 
        jitter=False,
        return_idx=False,
        trim=False,
        normalize=None
        ):

    csv_path = os.path.join(data_path, 'data_split_js.csv')
    images_dir = data_path

    # 데이터프레임 로드
    df = pd.read_csv(csv_path)

    # 데이터셋 분리
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    basic_transforms = []
    train_transform, test_transform = make_transforms(basic_transforms, macenko=macenko, resize=resize, crop=crop, hflip=hflip, vflip=vflip, rotation=rotation, jitter=jitter, normalize=normalize)

    # 데이터셋 생성
    train_dataset = MSI_CRC_Dataset(train_df, images_dir, macenko=macenko, transform=train_transform, shuffle=shuffle, return_idx=return_idx, trim=trim)
    val_dataset = MSI_CRC_Dataset(val_df, images_dir, macenko=macenko, transform=test_transform, return_idx=return_idx, trim=trim)
    test_dataset = MSI_CRC_Dataset(test_df, images_dir, macenko=macenko, transform=test_transform, return_idx=return_idx, trim=trim)
    print(f'MSI_CRC tran/val/test size: {len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}')
    print(f'MSI_CRC total size: {len(train_dataset) + len(val_dataset) + len(test_dataset)}')

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, \
                            worker_init_fn=train_dataset.worker_init_fn, pin_memory=True, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, \
                            worker_init_fn=val_dataset.worker_init_fn, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, \
                             worker_init_fn=test_dataset.worker_init_fn, pin_memory=True, drop_last=False)
    
    return train_loader, val_loader, test_loader


#############################################################################################
class MSI_STAD_Dataset(MSI_CRC_Dataset):
    '''
    MSI_CRC랑 똑같음
    '''
    pass
    
def MSI_STAD_dataloader(
        data_path, 
        batch_size=128, 
        num_workers=4, 
        shuffle=False, 
        drop_last=True, 
        macenko=False, 
        resize=None, 
        crop=None, 
        hflip=False, 
        vflip=False, 
        rotation=False, 
        jitter=False,
        return_idx=False,
        trim=False,
        normalize=None
        ):
    '''
    MSI_CRC랑 똑같음
    '''

    csv_path = os.path.join(data_path, 'data_split_js.csv')
    images_dir = data_path

    # 데이터프레임 로드
    df = pd.read_csv(csv_path)

    # 데이터셋 분리
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    basic_transforms = []
    train_transform, test_transform = make_transforms(basic_transforms, macenko=macenko, resize=resize, crop=crop, hflip=hflip, vflip=vflip, rotation=rotation, jitter=jitter, normalize=normalize)

    # 데이터셋 생성
    train_dataset = MSI_STAD_Dataset(train_df, images_dir, macenko=macenko, transform=train_transform, shuffle=shuffle, return_idx=return_idx, trim=trim)
    val_dataset = MSI_STAD_Dataset(val_df, images_dir, macenko=macenko, transform=test_transform, return_idx=return_idx, trim=trim)
    test_dataset = MSI_STAD_Dataset(test_df, images_dir, macenko=macenko, transform=test_transform, return_idx=return_idx, trim=trim)
    print(f'MSI_STAD tran/val/test size: {len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}')
    print(f'MSI_STAD total size: {len(train_dataset) + len(val_dataset) + len(test_dataset)}')

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, \
                            worker_init_fn=train_dataset.worker_init_fn, pin_memory=True, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, \
                            worker_init_fn=val_dataset.worker_init_fn, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, \
                             worker_init_fn=test_dataset.worker_init_fn, pin_memory=True, drop_last=False)
    
    return train_loader, val_loader, test_loader


#############################################################################################
# Main dataloader creation function
def make_dataloader(
        data, 
        shared='745ab7',
        path=None, 
        batch_size=128, 
        num_workers=4, 
        shuffle=True, 
        drop_last=True, 
        macenko=False, 
        resize=None, 
        crop=None, 
        hflip=True, 
        vflip=True, 
        rotation=True, 
        jitter=True, 
        return_idx=False, 
        trim=False,
        normalize=None,
        **kwargs):
    '''
    지금 구현상 persistant worker쓰면 매 epoch마다 shuffle이 안됨
    매 epoch마다 worker process 다시 만들어야 됨
    '''
    if path is None:
        base_path = f'/data/{shared}/shared/j.jang/pathai/data/downstream_data'
    else:
        base_path = path
    if macenko:
        normalizer = macenko_normalizer(shared)
    else:
        normalizer = False

    data_loader_params = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'shuffle': shuffle,
        'drop_last': drop_last,
        'macenko': normalizer,
        'resize': resize,
        'crop': crop,
        'hflip': hflip,
        'vflip': vflip,
        'rotation': rotation,
        'jitter': jitter,
        'return_idx': return_idx,
        'trim': trim,
        'normalize': normalize,
        **kwargs
    }

    match data.lower():
        case 'pcam' | 'patchcamelyon':
            data_path = os.path.join(base_path, 'PatchCamelyon')
            return PCAM_dataloader(data_path, **data_loader_params)
        case 'mhist':
            data_path = os.path.join(base_path, 'MHIST')
            return MHIST_dataloader(data_path, **data_loader_params)
        case 'crc100k' | 'crc-100k':
            data_path = os.path.join(base_path, 'CRC100k')
            return CRC100k_dataloader(data_path, **data_loader_params)
        case 'til det' | 'til-det' | 'til_det':
            data_path = os.path.join(base_path, 'TIL-Det/TCGA-TILs')
            return TIL_Det_dataloader(data_path, **data_loader_params)
        case 'msi crc' | 'msi_crc' | 'msi-crc':
            data_path = os.path.join(base_path, 'MSI_CRC')
            return MSI_CRC_dataloader(data_path, **data_loader_params)
        case 'msi stad' | 'msi_stad':
            data_path = os.path.join(base_path, 'MSI_STAD')
            return MSI_STAD_dataloader(data_path, **data_loader_params)
        case _:
            raise ValueError(f"Unknown dataset: {data}")



def get_args_parser():
    parser = argparse.ArgumentParser('DINO_downstream', add_help=False)
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser


def main(args): 
    init_distributed_mode(args)

    dataset_name = 'PCAM'
    # dataset_name = 'MHIST'
    # dataset_name = 'CRC100k'
    # dataset_name = 'TIL-Det'
    # dataset_name = 'MSI_CRC'
    # dataset_name = 'MSI_STAD'
    train_loader, val_loader, test_loader = make_dataloader(dataset_name, batch_size=64, num_workers=2, shuffle=True, drop_last=True, macenko=True)

    # 훈련 루프
    for epoch in range(2):
        i=0
        for images, labels in train_loader:
            # print(images.size(), labels.size())
            # print(labels[:9])
            print(images.size())
            # print(images[0][0][0][:9])

            # import matplotlib.pyplot as plt
            # plt.imshow(images[0])  # Convert image tensor to numpy array for display
            # plt.show()

            i+=1
            if i == 10:  
                break

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser('DINO_downstream', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)