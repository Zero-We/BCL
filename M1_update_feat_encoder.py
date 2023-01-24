import os
import numpy as np
import argparse
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from models.resnet_custom import resnet50_baseline
import torchvision.transforms as transforms
import openslide
from sklearn.metrics import roc_curve, auc
import pickle
import time
from Early_Stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau

parser = argparse.ArgumentParser(description='Updating the feature extractor and patch classifier')
parser.add_argument('--results_dir', type=str, default='', help='directory to save results')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--nepochs', type=int, default=200, help='the maxium number of epochs to train')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--slide_dir', type=str, default='', help='path to save wsi')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--device_ids', type=int, nargs='+', default=[0, 1])
parser.add_argument('--last_model', type=str, default=None)
parser.add_argument('--round', type=int, default=1)
global args, best_acc
args = parser.parse_args()
torch.cuda.set_device(args.device)
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

np.random.seed(24)
torch.manual_seed(24)
torch.cuda.manual_seed(24)
torch.cuda.manual_seed_all(24)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def main():
    best_acc = 0
    model = Model(input_dim=1024)

    start = time.time()
    if args.last_model:
        ch = torch.load(args.last_model, map_location='cpu')
        model.load_state_dict(ch, strict=False)
        print("Successfully load weight.")


    model.to(device)
    model = nn.DataParallel(model, device_ids=args.device_ids)

    # normalization
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.ToTensor(),
        normalize])

    test_trans = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    model_save_dir = args.results_dir
    with open(os.path.join(model_save_dir, f't{args.round}_pseudo_label.pkl'), 'rb') as f:
        obj = pickle.load(f)
    train_dset_patch = obj['train_dset_patch']
    val_dset_patch = obj['val_dset_patch']

    train_dset = Dataset(split=train_dset_patch, transform=trans)
    # train_dset = ImageDataset(split=train_dset_patch, transform=trans)  ## if save in .jpg format
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)
    val_dset = Dataset(split=val_dset_patch, transform=test_trans)
    # val_dset = ImageDataset(split=val_dset_patch, transform=test_trans)  ## if save in .jpg format
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # optimization
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    cudnn.benchmark = True

    model_save_path = os.path.join(model_save_dir, f't{args.round}_feature_extractor.pth')
    early_stopping = EarlyStopping(model_path=model_save_path,
                                   patience=10, verbose=True)


    for epoch in range(args.nepochs):
        train_loss, train_err = train(epoch, model, train_loader, criterion, optimizer)
        auc, val_err, val_loss = test(epoch, model, val_loader, criterion)
        print('Validating - Epoch: [{}/{}]\tLoss: {:.4f}\tACC: {:.4f}\tAUC: {:.4f}\t'.format(epoch + 1, args.nepochs,
                                                                                             val_loss,
                                                                                             1 - val_err, auc))

        early_stopping(epoch, val_loss, best_acc, model.module)
        if early_stopping.early_stop:
            print('Early Stopping')
            break
        print('\r')

    ch = torch.load(model_save_path, map_location='cpu')
    model.module.load_state_dict(ch, strict=False)
    auc, test_err, test_loss = test(0, model, val_loader, criterion)
    end = time.time()
    print('use time: ', end - start)
    print('Test\tLoss: {:.4f}\tACC: {:.4f}\tAUC: {:.4f}\t'.format(test_loss, 1 - test_err, auc))


def train(epoch, model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.
    running_err = 0.
    for i, (img, label) in enumerate(loader):
        optimizer.zero_grad()
        img = img.cuda()
        label = label.cuda()
        probs, _ = model(img)
        loss = criterion(probs, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * label.size(0)
        ## binary classification
        err, fps, fns = errors(probs.detach(), label.cpu())
        running_err += err
    running_loss = running_loss / len(loader.dataset)
    running_err = running_err / len(loader.dataset)
    print('Training - Epoch: [{}/{}]\tLoss: {:.4f}\tAcc: {:.4f}\t'.format(epoch + 1, args.nepochs, running_loss,
                                                                          1 - running_err))
    return running_loss, running_err


def test(epoch, model, loader, criterion):
    model.eval()
    running_loss = 0.
    running_err = 0.
    probs = []
    labels = []

    with torch.no_grad():
        for i, (img, label) in enumerate(loader):
            img = img.cuda()
            label = label.cuda()
            prob, _ = model(img)
            loss = criterion(prob, label)

            running_loss += loss.item() * label.size(0)
            err, fps, fns = errors(prob.detach(), label.cpu())
            running_err += err
            probs.extend(prob.detach()[:, 1].tolist())
            labels.extend(label.detach().tolist())
    fpr, tpr, thresholds = roc_curve(y_true=labels, y_score=probs, pos_label=1)
    roc_auc = auc(fpr, tpr)
    running_loss = running_loss / len(loader.dataset)
    running_err = running_err / len(loader.dataset)
    return roc_auc, running_err, running_loss


def errors(output, target):
    _, pred = output.topk(1, 1, True, True)  # return the max index of output
    pred = pred.squeeze().cpu().numpy()
    real = target.numpy()
    neq = pred != real
    err = float(neq.sum())
    fps = float(np.logical_and(pred == 1, neq).sum())
    fns = float(np.logical_and(pred == 0, neq).sum())
    return err, fps, fns


class Dataset(data.Dataset):
    def __init__(self, split, transform=None):
        self.img_id = split.keys()
        self.slides = []
        for img_id in self.img_id:
            flag = img_id.split('_')[0]
            slide_path = os.path.join(args.slide_dir, flag, img_id + '.tif')
            self.slides.append(openslide.OpenSlide(slide_path))

        self.labels, self.grids, self.slideIDX = self.get_data(split)
        self.transform = transform

    def get_data(self, split):
        labels = []
        grids = []
        slideIDX = []
        for i, (img_id, data) in enumerate(split.items()):
            coords = data['coords']
            label = data['labels']
            labels.extend(label)
            grids.extend(coords)
            slideIDX.extend([i] * len(label))
        return labels, grids, slideIDX

    def __getitem__(self, index):
        coord = self.grids[index]
        slide = self.slides[self.slideIDX[index]]
        label = self.labels[index]
        img = slide.read_region(coord, 1, (256, 256)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.labels)


class ImageDataset(data.Dataset):
    def __init__(self, split, transform=None):
        self.img_id = list(split.keys())
        self.patch_paths, self.labels = self.get_data(split)
        self.transform = transform

    def get_data(self, split):
        labels = []
        patch_paths = []
        for i, (img_id, data) in enumerate(split.items()):
            coords = data['coords']
            label = data['labels']
            labels.extend(label)
            patch_paths.extend(coords)
        return patch_paths, labels

    def __getitem__(self, index):
        img = Image.open(self.patch_paths[index]).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.labels)


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


if __name__ == '__main__':
    main()