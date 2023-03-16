import h5py
import pickle
import argparse
import os
import torch
import numpy as np
import pandas as pd
import glob

parser = argparse.ArgumentParser(description='Pseudo labeling')
parser.add_argument('--results_dir', type=str, default='', help='directory to save results')
parser.add_argument('--data_dir', type=str, default='', help='directory to save data')
parser.add_argument('--dset', type=str, default='camelyon')
parser.add_argument('--label_csv', type=str, default=None)
parser.add_argument('--round', type=int, default=1)
args = parser.parse_args()

def prepare_data(df):
    df_case_id = df['case_id'].tolist()
    df_slide_id = df['slide_id'].tolist()
    # df_code = df['oncotree_code'].tolist()
    df_label = df['label'].tolist()

    df_slide_id = [_slide_id.rstrip('.svs') for _slide_id in df_slide_id]
    slide_to_label = dict(zip(df_slide_id, df_label))
    return slide_to_label


model_save_dir = args.results_dir
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
with open(os.path.join(model_save_dir, f't{args.round}_primary_attn.pkl'), 'rb') as f:
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


patch_num = 10 * args.round
## training set
topmax_tumor = patch_num
topmax_normal = patch_num
topmin = patch_num
train_dset_patch = {}
for img_id, attn in train_attns.items():
    if img_id.startswith('tumor'):
        target = 1
    else:
        target = 0

    ## bal
    if args.round == 1:
        attn = torch.from_numpy(attn).squeeze(0)
        score = attn
    else:
        attn = torch.from_numpy(attn).squeeze(0)
        preds = torch.from_numpy(train_preds[img_id])
        preds = torch.transpose(preds, 1, 0)
        preds = preds[target]
        attn = (attn - torch.min(attn)) / (torch.max(attn) - torch.min(attn))
        score = preds * attn

    h5py_path = os.path.join(args.data_dir, 'patches', img_id + '.h5')
    file = h5py.File(h5py_path, 'r')
    coord_dset = file['coords']
    coords = np.array(coord_dset[:])

    ### if save image in .jpg format
    # patch_list = glob.glob(os.path.join('/mnt/MedImg/CAMELYON16/patch/clam20_256/img/', img_id, '*.jpg'))
    # patch_list = np.array(patch_list)

    if args.dset == 'camelyon':
        _, topmax_id = torch.topk(score, k=topmax_tumor, dim=0)
        _, topmin_id = torch.topk(-score, k=topmin, dim=0)
        label = [target] * topmax_id.size(0) + [0] * topmin_id.size(0)

        idx = topmax_id.tolist() + topmin_id.tolist()
        topmax_id = topmax_id.numpy()
        topmin_id = topmin_id.numpy()

        topmax_coords = coords[topmax_id].tolist()
        topmin_coords = coords[topmin_id].tolist()

        ### image path list
        # topmax_coords = patch_list[topmax_id].tolist()
        # topmin_coords = patch_list[topmin_id].tolist()

        select_coords = topmax_coords + topmin_coords
        # select_coords = topmax_coords
    train_dset_patch[img_id] = {'coords': select_coords, 'labels': label, 'idx': idx}
    #print('Finish ', img_id)


## validation set
topmax_tumor = patch_num
topmax_normal = patch_num
topmin = patch_num
val_dset_patch = {}
for img_id, attn in val_attns.items():
    if img_id.startswith('tumor'):
        target = 1
    else:
        target = 0

    if args.round == 1:
        attn = torch.from_numpy(attn).squeeze(0)
        score = attn
    else:
        attn = torch.from_numpy(attn).squeeze(0)
        preds = torch.from_numpy(val_preds[img_id])
        preds = torch.transpose(preds, 1, 0)
        preds = preds[target]
        attn = (attn - torch.min(attn)) / (torch.max(attn) - torch.min(attn))
        score = preds * attn

    h5py_path = os.path.join(args.data_dir, 'patches', img_id + '.h5')
    file = h5py.File(h5py_path, 'r')
    coord_dset = file['coords']
    coords = np.array(coord_dset[:])


    ### if save image in .jpg format
    # patch_list = glob.glob(os.path.join('/mnt/MedImg/CAMELYON16/patch/clam20_256/img/', img_id, '*.jpg'))
    # patch_list = np.array(patch_list)

    if args.dset == 'camelyon':
        _, topmax_id = torch.topk(score, k=topmax_tumor, dim=0)
        _, topmin_id = torch.topk(-score, k=topmin, dim=0)
        label = [target] * topmax_id.size(0) + [0] * topmin_id.size(0)

        idx = topmax_id.tolist() + topmin_id.tolist()
        topmax_id = topmax_id.numpy()
        topmin_id = topmin_id.numpy()

        topmax_coords = coords[topmax_id].tolist()
        topmin_coords = coords[topmin_id].tolist()

        ### image path list
        # topmax_coords = patch_list[topmax_id].tolist()
        # topmin_coords = patch_list[topmin_id].tolist()

        select_coords = topmax_coords + topmin_coords
    val_dset_patch[img_id] = {'coords': select_coords, 'labels': label, 'idx': idx}


new_obj = {
    'train_dset_patch': train_dset_patch,
    'val_dset_patch': val_dset_patch
}

with open(os.path.join(model_save_dir, f't{args.round}_pseudo_label.pkl'), 'wb') as f:
    pickle.dump(new_obj, f)
print('Finish')
