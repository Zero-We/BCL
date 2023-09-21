import torch
import glob
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])


class MyDataset(Dataset):
    def __init__(self, feat_dir,  transform=None, split=None, train=True, dset='camelyon'):
        self.train = train
        self.transform = transform
        self.feat_dir = feat_dir
        self.img_id = list(split.keys())
        self.label = list(split.values())
        self.dset = dset
        self.feat_files = self.get_feat_file()

    def get_feat_file(self):
        feat_files = {}
        for id in self.img_id:
            if self.dset == 'camelyon':
                feat_files[id] = [os.path.join(self.feat_dir, id+'.pt')]
            else:
                feat_file = glob.glob(os.path.join(self.feat_dir, id+'*'))
                feat_files[id] = feat_file
        return feat_files

    def __getitem__(self, idx):
        img_name = self.img_id[idx]
        feat_files = self.feat_files[img_name]
        resnet_feats = torch.Tensor()
        for feat_file in feat_files:
            resnet_feat =  torch.load(feat_file, map_location='cpu')
            resnet_feats = torch.cat((resnet_feats, resnet_feat), dim=0)
        feats = resnet_feats
        target = self.label[idx]
        sample = {'img_id': img_name, 'feat': feats, 'target': target}
        return sample

    def __len__(self):
        return len(self.img_id)


