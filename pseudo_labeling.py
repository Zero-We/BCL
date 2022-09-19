import h5py
import pickle
import argparse
import os
import torch
import numpy as np
import pandas as pd
import glob

parser = argparse.ArgumentParser(description='DeepAttnMISL')
parser.add_argument('--results_dir', type=str, default="result/CAMELYON", help='Results directory (default: ./result)')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--dset', type=str, default='camelyon')
parser.add_argument('--label_csv', type=str, default=None)
args = parser.parse_args()

def prepare_data(df):
    df_case_id = df['case_id'].tolist()
    df_slide_id = df['slide_id'].tolist()
    df_label = df['label'].tolist()

    df_slide_id = [_slide_id.rstrip('.svs') for _slide_id in df_slide_id]
    slide_to_label = dict(zip(df_slide_id, df_label))
    return slide_to_label


model_save_path = args.results_dir
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
with open(os.path.join(model_save_path, 't0_primary_attn.pkl'), 'rb') as f:
    obj = pickle.load(f)
train_attns = obj['train_attns']
train_preds = obj['train_preds']
val_attns = obj['val_attns']
val_preds = obj['val_preds']
test_attns = obj['test_attns']
test_preds = obj['test_preds']


## load label_csv
if args.label_csv:
    df = pd.read_csv(args.label_csv)
    slide_to_label = prepare_data(df)


train_dset_patch = {}
for img_id, attn in train_attns.items():
    # labeling train dset patch
    topmax_tumor = 10
    topmax_normal = 10
    topmin = 10
    if img_id.startswith('tumor'):
        target = 1
    else:
        target = 0
    attn = torch.from_numpy(attn).squeeze(0)
    preds = torch.from_numpy(train_preds[img_id])
    preds = torch.transpose(preds, 1, 0)
    
    ## binary class
    h5py_path = os.path.join(args.data_dir, 'patches', img_id + '.h5')
    file = h5py.File(h5py_path, 'r')
    coord_dset = file['coords']
    coords = np.array(coord_dset[:])
    attn = (attn - torch.min(attn)) / (torch.max(attn) - torch.min(attn))
    preds = preds[target]
    score = preds * attn
    
    if args.dset == 'camelyon':
        topmax_tumor = score.size(0) if score.size(0) < topmax_tumor else topmax_tumor
        topmin = score.size(0) if score.size(0) < topmin else topmin
        _, topmax_id = torch.topk(score, k=topmax_tumor, dim=0)
        _, topmin_id = torch.topk(-score, k=topmin, dim=0)
        label = [target] * topmax_id.size(0) + [0] * topmin_id.size(0)
        idx = topmax_id.tolist() + topmin_id.tolist()
        topmax_id = topmax_id.numpy()
        topmin_id = topmin_id.numpy()
        topmax_coords = coords[topmax_id].tolist()
        topmin_coords = coords[topmin_id].tolist()
        
        select_coords = topmax_coords + topmin_coords
    train_dset_patch[img_id] = {'coords': select_coords, 'labels': label, 'idx':idx}
    print('Finish ', img_id)

val_dset_patch = {}
for img_id, attn in val_attns.items():
    # labeling val dset patch
    topmax_tumor = 10
    topmax_normal = 10
    topmin = 10
    if img_id.startswith('tumor'):
        target = 1
    else:
        target = 0
    attn = torch.from_numpy(attn).squeeze(0)
    preds = torch.from_numpy(val_preds[img_id])
    preds = torch.transpose(preds, 1, 0)

    h5py_path = os.path.join(args.data_dir, 'patches', img_id + '.h5')
    file = h5py.File(h5py_path, 'r')
    coord_dset = file['coords']
    coords = np.array(coord_dset[:])
    
    attn = (attn - torch.min(attn)) / (torch.max(attn) - torch.min(attn))
    preds = preds[target]
    score = preds * attn


    if args.dset == 'camelyon':
        topmax_tumor = score.size(0) if score.size(0) < topmax_tumor else topmax_tumor
        topmin = score.size(0) if score.size(0) < topmin else topmin
        _, topmax_id = torch.topk(score, k=topmax_tumor, dim=0)
        _, topmin_id = torch.topk(-score, k=topmin, dim=0)
        label = [target] * topmax_id.size(0) + [0] * topmin_id.size(0)
        print(img_id, ': ', label)
        idx = topmax_id.tolist() + topmin_id.tolist()
        topmax_id = topmax_id.numpy()
        topmin_id = topmin_id.numpy()
        topmax_coords = coords[topmax_id].tolist()
        topmin_coords = coords[topmin_id].tolist()
        
        select_coords = topmax_coords + topmin_coords
    val_dset_patch[img_id] = {'coords': select_coords, 'labels': label, 'idx':idx}
    print('Finish ', img_id)


new_obj = {
    'train_dset_patch': train_dset_patch,
    'val_dset_patch': val_dset_patch
}

with open(os.path.join(model_save_path, 't0_pseudo_label.pkl'), 'wb') as f:
    pickle.dump(new_obj, f)
print('Finish')
