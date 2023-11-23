import os
from PIL import Image, ImageDraw, ImageFont
import h5py
from torchvision import transforms, datasets, models
import torch.backends.cudnn as cudnn
import cv2
import numpy as np
import openslide
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import xml.etree.ElementTree as et
import pandas as pd
import torch

parser = argparse.ArgumentParser(description='Generate WSI attention maps')
parser.add_argument('--output_dir', type=str, default='', help='output directory')
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--anno_path', type=str, default='', help='directory of annotations')
parser.add_argument('--results_dir', type=str, default='', help='Results directory')
parser.add_argument('--slide_dir', type=str, default='', help='directory to save WSI')
parser.add_argument('--data_dir', type=str, default='')

global args
args = parser.parse_args()


def main():
    model_save_path = args.results_dir
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    with open(os.path.join(model_save_path, 'vis_t0_primary_attn.pkl'), 'rb') as f:
        obj = pickle.load(f)
    train_attns = obj['train_attns']
    val_attns = obj['val_attns']
    test_attns = obj['test_attns']


    cudnn.benchmark = True
    draw_level = 8
    final_level = 4
    final_level2 = 7

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ## Draw Trainset Heatmap
    anno_name = [anno.split('.')[0] for anno in os.listdir(args.anno_path)]
    anno_name.sort()

    for i, (img_id, attn) in enumerate(test_attns.items()):
        
        if img_id not in ['test_021', 'test_090', 'test_122']:
            continue
        
        slidepath = os.path.join(args.slide_dir, img_id.split('_')[0], img_id+'.tif')
        slide = openslide.OpenSlide(slidepath)

        ## load grid
        h5py_path = os.path.join(args.data_dir, 'patches', img_id+'.h5')
        file = h5py.File(h5py_path, 'r')
        grid = file['coords'][:]
        #coords = np.array(coord_dset[:])

        size = slide.level_dimensions[final_level]
        final_size = slide.level_dimensions[final_level2]
        downsample = slide.level_downsamples[draw_level]
        draw_size = slide.level_dimensions[draw_level]

        wsi_ori = slide.get_thumbnail(size)
        wsi_ori = wsi_ori.resize(size)
        wsi_ori = np.array(wsi_ori)
        #wsi_ori = cv2.cvtColor(wsi_ori, cv2.COLOR_BGR2RGB)
        wsi_mask = cv2.cvtColor(wsi_ori, cv2.COLOR_BGR2GRAY)
        _, wsi_mask = cv2.threshold(wsi_mask, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        wsi_mask = 1 - wsi_mask

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        wsi_mask = cv2.morphologyEx(wsi_mask, cv2.MORPH_CLOSE, kernel)

        heatmap = np.zeros((draw_size[1], draw_size[0]))
        heatmap, mask = generate_heatmap(heatmap, grid, downsample, attn[0,:])
        cmin = np.min(heatmap)
        cmax = np.max(heatmap)

        cam_img = np.uint8(255 * heatmap)

        #colorheatmap = cv2.applyColorMap(cam_img, get_mpl_colormap('bwr_r'))
        colorheatmap = cv2.applyColorMap(cam_img, get_mpl_colormap('jet_r'))
        colorheatmap[mask] = np.array([255, 255, 255])
        colorheatmap = cv2.resize(colorheatmap, size)
        colorheatmap[wsi_mask==0] = np.array([255, 255, 255])
        colorheatmap = cv2.GaussianBlur(colorheatmap, (5, 5), 5.0)
        result = 0.4 * colorheatmap + wsi_ori * 0.6
        result = np.uint8(result)


        ## draw original image
        result = wsi_ori

        if img_id in anno_name:
            ## Loading annotation
            annotations = convert_xml_df(os.path.join(args.anno_path, img_id + '.xml'), slide.level_downsamples[final_level])
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
            cv2.drawContours(result, coxy, -1, (255, 255, 0), 30)


        result = cv2.resize(result, final_size)
        result = cv2.cvtColor(np.uint8(result), cv2.COLOR_BGR2RGB)

        cv2.imwrite(os.path.join(args.output_dir, f"{img_id}.png"), result)
        
        print("WSI :", img_id)
        #break

def to_percentiles(scores):
    from scipy.stats import rankdata
    scores = rankdata(scores, 'min')/len(scores)   
    return scores

def generate_heatmap(heatmap, grid, downsample, attention_weight):
    for i in range(len(grid)):
        grid = np.array(grid, dtype=np.int)
        coor = get_draw_coor([grid[i,0], grid[i,1]], downsample)
        heatmap[coor[1]:coor[3], coor[0]:coor[2]] = attention_weight[i]
    mask = heatmap == 0
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    return heatmap, mask

def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)
    sm = plt.cm.ScalarMappable(cmap=cmap)
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
            X_coord = X_coord / downsample
            Y_coord = float(coordinate.attrib.get('Y'))
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


if __name__ == "__main__":
    main()






