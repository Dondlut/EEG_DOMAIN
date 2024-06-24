import os
import numpy as np
import h5py
import torch
from random import shuffle
import time
from constant_22 import dev_id, test_id
from torch.utils.data import DataLoader, Dataset


class SourceDataset(Dataset):

    def __init__(self):
        raw_data_dir = r''
        self.nums_dict = {0: 0, 1: 0, 2: 0, 3: 0}

        self.X_source = []
        self.y_label = []
        self.y_domain = []
        file_list = os.listdir(raw_data_dir)
        for file in file_list:
            patient_id = file.split("_")[0]  # 00000002_s001_t000.edf_0_1_0.h5
            # 验证集病人数据为target domain
            if patient_id in dev_id or patient_id in test_id:
                continue
            label = int(file.split("_")[-1].split(".")[0])
            fileFullPath = os.path.join(raw_data_dir, file)
            self.X_source.append(fileFullPath)
            self.y_domain.append(0)  # source domain label
            self.y_label.append(label)
            self.nums_dict[label] += 1

        length = len(self.X_source)
        print('source domain：', self.nums_dict)
        self.size = length

    def __getitem__(self, item):

        sz_file_path_1 = self.X_source[item]
        with h5py.File(sz_file_path_1, 'r') as hf:
            eeg_clip = hf['clip'][()]
        x = torch.from_numpy(eeg_clip)
        y_label = torch.tensor(self.y_label[item])
        y_domain = torch.tensor(self.y_domain[item])

        return x, y_label, y_domain

    def __len__(self):
        return self.size


class TargetDataset(Dataset):

    def __init__(self):
        raw_data_dir = r''

        self.nums_dict = {0: 0, 1: 0, 2: 0, 3: 0}

        self.X_target = []
        self.y_domain = []
        self.y_label = []
        file_list = os.listdir(raw_data_dir)
        for file in file_list:
            patient_id = file.split("_")[0]  # 00000002_s001_t000.edf_0_1_0.h5
            # 验证集病人
            if patient_id not in dev_id:
                continue
            fileFullPath = os.path.join(raw_data_dir, file)
            label = int(file.split("_")[-1].split(".")[0])
            self.X_target.append(fileFullPath)
            self.y_label.append(label)
            self.y_domain.append(1)  # target domain label
            self.nums_dict[label] += 1
        print('target domain：', self.nums_dict)
        length = len(self.X_target)
        self.size = length

    def __getitem__(self, item):

        sz_file_path_1 = self.X_target[item]
        with h5py.File(sz_file_path_1, 'r') as hf:
            eeg_clip = hf['clip'][()]
        x1 = torch.from_numpy(eeg_clip)
        y_label = torch.tensor(self.y_label[item])
        y_domain = torch.tensor(self.y_domain[item])

        return x1, y_label, y_domain

    def __len__(self):
        return self.size


class TestDataset(Dataset):

    def __init__(self):
        raw_data_dir = r''

        self.nums_dict = {0: 0, 1: 0, 2: 0, 3: 0}

        self.X_test = []
        self.patient = []
        self.y_label = []
        file_list = os.listdir(raw_data_dir)
        for file in file_list:
            patient_id = file.split("_")[0]  # 00000002_s001_t000.edf_0_1_0.h5
            # 测试集病人
            if patient_id not in test_id:
                continue
            fileFullPath = os.path.join(raw_data_dir, file)
            label = int(file.split("_")[-1].split(".")[0])
            self.X_test.append(fileFullPath)
            self.y_label.append(label)
            self.patient.append(patient_id)
            self.nums_dict[label] += 1
        print('test domain：', self.nums_dict)
        length = len(self.X_test)
        self.size = length

    def __getitem__(self, item):

        sz_file_path_1 = self.X_test[item]
        with h5py.File(sz_file_path_1, 'r') as hf:
            eeg_clip = hf['clip'][()]
        x1 = torch.from_numpy(eeg_clip)
        y_label = torch.tensor(self.y_label[item])
        patient = self.patient[item]

        return x1, y_label, patient, sz_file_path_1

    def __len__(self):
        return self.size


source_loader = DataLoader(SourceDataset(), batch_size=60, shuffle=True, num_workers=0, drop_last=True)
print(len(source_loader))
target_loader = DataLoader(TargetDataset(), batch_size=19, shuffle=True, num_workers=0, drop_last=True)
print(len(target_loader))
test_loader = DataLoader(TestDataset(), batch_size=8, shuffle=True, num_workers=0)
