import os
from PIL import Image, ImageDraw, ImageFont
import h5py
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torchvision import transforms, datasets, models
import torch.nn.functional as F
import cv2
import numpy as np
import openslide
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import xml.etree.ElementTree as et
import pandas as pd
from models.resnet_custom import resnet50_baseline

parser = argparse.ArgumentParser(description='Generate patch score maps')
parser.add_argument('--output_dir', type=str, default='', help='output directory')
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--anno_path', type=str, default='', help='directory of annotations')
parser.add_argument('--results_dir', type=str, default='', help='Results directory ')
parser.add_argument('--slide_dir', type=str, default='', help='directory to save WSI')
parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--last_model', type=str, default='')

global args, best_acc
args = parser.parse_args()
torch.cuda.set_device(args.device)
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')


def main():
    model_save_path = args.results_dir
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    model = Model(input_dim=1024)
    if args.last_model:
        ch = torch.load(args.last_model, map_location='cpu')
        print("Successfully load weight.")
        model.load_state_dict(ch)
    model.to(device)

    cudnn.benchmark = True
    draw_level = -1
    final_level = 4
    final_level2 = -2

    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    trans = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    slidenames = os.listdir(args.slide_dir)
    slidepaths = [os.path.join(args.slide_dir, slidename) for slidename in slidenames]

    for i, slidepath in enumerate(slidepaths):
        slidename = os.path.basename(slidepath).split('.')[0]
        
        slide = openslide.OpenSlide(slidepath)

        ## load grid
        h5py_path = os.path.join(args.data_dir, 'patches', slidename+'.h5')
        file = h5py.File(h5py_path, 'r')
        grid = file['coords'][:]

        
        dset = Dataset(i, len(slidepaths), slidepath, grid, transform=trans)
        #dset = ImageDataset(i, len(slidepaths), slidename, args.patch_dir, transform=trans)
        loader = torch.utils.data.DataLoader(
            dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)

        size = slide.level_dimensions[final_level]
        final_size = slide.level_dimensions[final_level2]
        downsample = slide.level_downsamples[draw_level]
        draw_size = slide.level_dimensions[draw_level]

        wsi_ori = slide.get_thumbnail(size)
        wsi_ori = wsi_ori.resize(size)
        wsi_ori = np.array(wsi_ori)

        wsi_mask = cv2.cvtColor(wsi_ori, cv2.COLOR_BGR2GRAY)
        _, wsi_mask = cv2.threshold(wsi_mask, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        wsi_mask = 1 - wsi_mask

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        wsi_mask = cv2.morphologyEx(wsi_mask, cv2.MORPH_CLOSE, kernel)


        heatmap = torch.zeros(size=(draw_size[1], draw_size[0]))

        heatmap, mask = generate_heatmap(loader, model, heatmap, downsample)
        cmin = np.min(heatmap)
        cmax = np.max(heatmap)

        cam_img = np.uint8(255 * heatmap)

        #colorheatmap = cv2.applyColorMap(cam_img, get_mpl_colormap('bwr_r'))
        colorheatmap = cv2.applyColorMap(cam_img, get_mpl_colormap('jet_r'))
        colorheatmap = cv2.GaussianBlur(colorheatmap, (5, 5), 5.0)
        colorheatmap[mask] = np.array([255, 255, 255])
        colorheatmap = cv2.resize(colorheatmap, size)
        colorheatmap[wsi_mask==0] = np.array([255, 255, 255])



        result = 0.3 * colorheatmap + wsi_ori * 0.7
        result = np.uint8(result)


        ## Draw annotations
        if slidename in anno_name:
            ## Loading annotation
            annotations = convert_xml_df(os.path.join(args.anno_path, slidename + '.xml'), slide.level_downsamples[final_level])
            final_list = Remove(annotations['Name'])

            coxy = [[] for x in range(len(final_list))]

            j = 0
            for n in final_list:
                newx = annotations[annotations['Name'] == n]['X']
                newy = annotations[annotations['Name'] == n]['Y']
                newxy = list(zip(newx, newy))
                coxy[j] = np.array(newxy, dtype=np.int32)
                j = j + 1

            ## Draw annotation
            cv2.drawContours(result, coxy, -1, (255, 255, 0), 8)


        result = cv2.resize(result, final_size)
        result = cv2.cvtColor(np.uint8(result), cv2.COLOR_BGR2RGB)

        cv2.imwrite(os.path.join(args.output_dir, f"{slidename}.png"), result)
        
        print("WSI :", slidename)
        #break

def to_percentiles(scores):
    from scipy.stats import rankdata
    scores = rankdata(scores, 'min')/len(scores)   
    return scores


def generate_heatmap(loader, model, heatmap, downsample):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            img, grid = data
            img = img.to(device)
            grid = np.array(grid, dtype=np.int)
            output, feature = model(img)
            output = F.softmax(output, dim=1)
            probs = output.detach()[:,1].cpu().numpy()
            for j in range(probs.shape[0]):
                coor = get_draw_coor([grid[j,0], grid[j,1]], downsample)
                heatmap[coor[1]:coor[3], coor[0]:coor[2]] = probs[j].item()
    heatmap = heatmap.numpy()
    mask = heatmap == 0
    return heatmap, mask
    
    
def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)
    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)
    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:,2::-1]
    return color_range.reshape(256, 1, 3)

def get_draw_coor(grid, downsample):
    coor = [grid[0], grid[1], grid[0] + args.patch_size*2, grid[1] + args.patch_size*2]
    coor = [int(_coor / downsample) for _coor in coor]
    return coor


def convert_xml_df(file, downsample):
    parseXML = et.parse(file)
    root = parseXML.getroot()
    dfcols = ['Name', 'Order', 'X', 'Y']
    df_xml = pd.DataFrame(columns=dfcols)
    for child in root.iter('Annotation'):
        for coordinate in child.iter('Coordinate'):
            Name = child.attrib.get('Name')
            Order = coordinate.attrib.get('Order')
            X_coord = float(coordinate.attrib.get('X'))
            # X_coord = X_coord - 30000
            X_coord = X_coord / downsample
            Y_coord = float(coordinate.attrib.get('Y'))
            # Y_coord = Y_coord - 155000
            Y_coord = Y_coord / downsample
            df_xml = df_xml.append(pd.Series([Name, Order, X_coord, Y_coord], index=dfcols), ignore_index=True)
            df_xml = pd.DataFrame(df_xml)
    return (df_xml)

def Remove(duplicate):
    final_list = []
    for num in duplicate:
        if num not in final_list:
            final_list.append(num)
    return final_list

class Dataset(data.Dataset):
    def __init__(self, idx, total, slidepath, grid, transform=None):
        #Flatten grid
        self.slidepath = slidepath
        self.grid = grid

        self.wsiname = os.path.basename(slidepath).split('.')[0]
        self.slide = openslide.OpenSlide(slidepath)
        print('WSI: {}|{}\tNumber of tiles: {}'.format(idx+1, total, len(self.grid)))
        self.transform = transform


    def __getitem__(self,index):
        coord = self.grid[index]
        img = self.slide.read_region(coord, 1, (256,256)).convert('RGB')
        width, height = int(coord[0]), int(coord[1])
        coor = torch.Tensor([width, height])
        # img.save(os.path.join(self.wsidir, f"{coord[0]}_{coord[1]}.jpg"))
        if self.transform is not None:
            img = self.transform(img)
        return img, coor

    def __len__(self):
        return len(self.grid)

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

if __name__ == "__main__":
    main()






