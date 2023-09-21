import sys
import os
import numpy as np
import argparse
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.utils.data as data
import glob
from models.resnet_custom import resnet50_baseline
import torchvision.transforms as transforms
import openslide
import h5py
import time
import threading
import math


parser = argparse.ArgumentParser(description='Extracting feature from updated feature encoder')
parser.add_argument('--feat_dir', type=str, default='', help='directory to save features')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--patch_dir', type=str, default='', help='path to save patch image')
parser.add_argument('--coord_dir', type=str, default='', help='path to save patch coordinate')
parser.add_argument('--last_model', type=str, default='')
parser.add_argument('--results_dir', type=str, default='', help='directory to save results')
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--device_ids', type=int, nargs='+', default=[0, 1])
parser.add_argument('--self_supervised', default=True, action='store_true')
parser.add_argument('--thread', type=int, default=0)
parser.add_argument('--round', type=int, default=1)


global args, best_acc
args = parser.parse_args()
torch.cuda.set_device(args.device)
devices = ['cuda:0', 'cuda:1', 'cuda:1', 'cuda:0']


normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
trans = transforms.Compose([
    transforms.ToTensor(),
    normalize])


def extract_feature(slidepaths, device, thread_index):
    begin = time.time()

    model = Model(input_dim=1024)
    model.to(device)

    if args.last_model:
        ch = torch.load(args.last_model, map_location='cpu')
        print("Successfully load weight.")
        model.load_state_dict(ch)

    for i, slidepath in enumerate(slidepaths):
        slidename = os.path.basename(slidepath).split('.')[0]

        ## h5py coord
        h5py_path = os.path.join(args.coord_dir, slidename + '.h5')
        file = h5py.File(h5py_path, 'r')
        coord_dset = file['coords']
        grid = coord_dset[:]

        dset = Dataset(i, len(slidepaths), slidepath, grid, thread_index, transform=trans)
        loader = torch.utils.data.DataLoader(
            dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)
        features = infer(loader, model, device, slidename)

        torch.save(features, os.path.join(args.feat_dir, f'{slidename}.pt'))
    end = time.time()
    print('Usetime: ', end - begin)


def main():
    model_save_dir = args.results_dir

    slidepaths = []
    slide_dirs = ["/mnt/MedImg/CAMELYON16/tumor/", "/mnt/MedImg/CAMELYON16/normal/", "/mnt/MedImg/CAMELYON16/test/"]
    for slide_dir in slide_dirs:
        paths = glob.glob(os.path.join(slide_dir, '*.tif'))
        paths = sorted(paths)
        slidepaths.extend(paths)

    if not os.path.exists(args.feat_dir):
        os.makedirs(args.feat_dir)

    ## multi-threads
    threads = []
    fold = 1
    fold_size = math.ceil(len(slidepaths) / fold)
    # for thread_index in range(fold):
    thread_index = args.thread
    if thread_index == fold - 1:
        paths = slidepaths[thread_index * fold_size:]
        arg = (slidepaths[thread_index * fold_size:], devices[thread_index], thread_index)
    else:
        paths = slidepaths[thread_index * fold_size:(thread_index + 1) * fold_size]
        arg = (slidepaths[thread_index * fold_size:(thread_index + 1) * fold_size], devices[thread_index], thread_index)
    extract_feature(paths, devices[thread_index], thread_index)


def infer(loader, model, device, slidename):
    model.eval()
    features = torch.Tensor()
    with torch.no_grad():
        for i, input in enumerate(loader):
            input = input.to(device)
            probs, feature = model(input)
            # probs = F.softmax(probs.detach(), dim=1)
            features = torch.cat((features, feature.cpu()), dim=0)
    return features.cpu()


class Model(nn.Module):
    def __init__(self, input_dim=1024):
        super(Model, self).__init__()
        self.backbone = resnet50_baseline(True)
        self.backbone.fc = nn.Linear(1024, 2)

        self.fc = nn.Linear(1024, 2)

    def forward(self, x):
        _, feat = self.backbone(x)
        prob = self.fc(feat)
        return prob, feat


class Dataset(data.Dataset):
    def __init__(self, idx, total, slidepath, grid, thread_index, transform=None):
        self.slidepath = slidepath
        self.grid = grid
        self.wsiname = os.path.basename(slidepath).split('.')[0]
        # self.wsiname = os.path.basename(slidepath).rstrip('.svs')
        # self.wsidir = os.path.join(args.patch_dir, self.wsiname)
        # if not os.path.exists(self.wsidir):
        #     os.makedirs(self.wsidir)
        self.slide = openslide.OpenSlide(slidepath)
        print('WSI: {}|{}\tNumber of tiles: {} for thread: {}'.format(idx + 1, total, len(self.grid), thread_index))
        self.transform = transform

    def __getitem__(self, index):
        coord = self.grid[index]
        ## if save in .jpg format
        # img_path = os.path.join(self.wsidir, f"{coord[0]}_{coord[1]}.jpg")
        # img = Image.open(img_path)
        img = self.slide.read_region(coord, 1, (256, 256)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.grid)


class ImageDataset(data.Dataset):
    def __init__(self, idx, total, slidename, image_dir, transform=None):
        self.image_dir = image_dir
        # self.image_paths = glob.glob(os.path.join(image_dir, slidename, slidename+'*'))
        self.image_dir = os.path.join(self.image_dir, slidename)
        self.image_paths = glob.glob(os.path.join(self.image_dir, '*.jpg'))

        print('WSI: {}|{}\t'.format(idx + 1, total))
        self.transform = transform

    def __getitem__(self, index):
        # img = Image.open(os.path.join(self.image_dir, self.image_paths[index])).convert('RGB')
        img = Image.open(self.image_paths[index])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.image_paths)


if __name__ == '__main__':
    main()