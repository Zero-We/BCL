import torch
import numpy as np
from dataset.dataloader import MyDataset
import torch.nn as nn
from models.cls_model import BAL_P, BAL_A
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import pandas as pd
import os
import argparse
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import pickle
import torch.nn.functional as F
from Early_Stopping import EarlyStopping


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser(description='Updating the MIL classifier')
parser.add_argument('--feat_dir', type=str, default='', help='directory to save features')
parser.add_argument('--split_dir', type=str, default='', help='directory to save train/val/test split')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--epochs', type=int, default=200, help='the maxium number of epochs to train')
parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
parser.add_argument('--k', type=int, default=1, help='number of folds')
parser.add_argument('--results_dir', type=str, default='', help='directory to save results')
parser.add_argument('--seed', type=int, default=10, help='random seed for reproducible experiment')
parser.add_argument('--encoding_size', type=int, default=1024)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--label_csv', type=str, default=None)
parser.add_argument('--round', type=int, default=0)


args = parser.parse_args()
torch.cuda.set_device(args.device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(24)
torch.manual_seed(24)
torch.cuda.manual_seed(24)
torch.cuda.manual_seed_all(24)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def prepare_data(df, case_id):
    df_case_id = df['case_id'].tolist()
    df_slide_id = df['slide_id'].tolist()
    # df_code = df['oncotree_code'].tolist()
    df_label = df['label'].tolist()

    slide_id = []
    label = []
    for case_id_ in case_id:
        idx = df_case_id.index(case_id_)
        slide_id.append(df_slide_id[idx].rstrip('.svs'))
        # label.append(CLASS2ID[args.dataset][df_code[idx]])
        label.append(df_label[idx])
    return slide_id, label


def return_splits(from_id, csv_path):
    if from_id:
        raise NotImplementedError
    else:
        assert csv_path
        split_df = pd.read_csv(csv_path)

        train_id = split_df['train'].tolist()
        val_id = split_df['val'].dropna().tolist()
        test_id = split_df['test'].dropna().tolist()
        if not args.label_csv:
            train_label = split_df['train_label'].tolist()
            val_label = split_df['val_label'].dropna().tolist()
            test_label = split_df['test_label'].dropna().tolist()
        else:
            df = pd.read_csv(args.label_csv)
            train_id, train_label = prepare_data(df, train_id)
            val_id, val_label = prepare_data(df, val_id)
            test_id, test_label = prepare_data(df, test_id)

        train_split = dict(zip(train_id, train_label))
        val_split = dict(zip(val_id, val_label))
        test_split = dict(zip(test_id, test_label))

    return train_split, val_split, test_split


def prediction(model, loader, criterion, testing=False):
    model.eval()

    total_loss = 0.
    total_wsi_loss = 0.
    logits = torch.Tensor().cuda()
    target_wsi = []
    attns = {}


    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(loader):
            img_id, X, target = sampled_batch['img_id'], sampled_batch['feat'], sampled_batch['target']
            input = X.cuda()
            target = target.type(torch.int64).cuda()

            logit, Y_prob, Y_hat, attn = model(input)
            target_wsi.append(target.item())
            logits = torch.cat((logits, logit), dim=0)
            attns[img_id[0]] = attn
            loss_cls_wsi = criterion(logit, target).item()
            loss = loss_cls_wsi
            total_loss += loss
            total_wsi_loss += loss_cls_wsi
    total_loss = total_loss / len(loader.dataset)
    total_wsi_loss = total_wsi_loss / len(loader.dataset)
    print("[Valid] Loss: {:.4f} WSI Loss :{:.4f}".format(total_loss, total_wsi_loss))

    output = F.softmax(logits, dim=1)
    score, pred = torch.max(output, dim=1)

    pred = pred.cpu().numpy()
    output = output.cpu().numpy()
    target_wsi = np.array(target_wsi)
    eq = np.equal(target_wsi, pred)
    acc = float(eq.sum()) / target_wsi.shape[0]

    fpr, tpr, thresh = roc_curve(y_true=target_wsi, y_score=output[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)

    print(confusion_matrix(target_wsi, pred))

    if not testing:
        print('[val]\t loss (all):{:.4f}'.format(total_loss),
              'WSI acc: {:.4f}\t WSI auc: {:.4f}\t'.format(acc, roc_auc))
    else:
        print('[testing]\t loss (nn):{:.4f}'.format(total_loss),
              'WSI acc: {:.4f}\t WSI auc: {:.4f}\t'.format(acc, roc_auc))
    torch.cuda.empty_cache()
    return total_loss, roc_auc, attns


def patch_prediction(model, loader, criterion, testing=False):
    model.eval()
    logits = torch.Tensor()
    target_wsi = []

    instance_preds = {}
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(loader):
            img_id, X, target = sampled_batch['img_id'], sampled_batch['feat'], sampled_batch['target']
            input = X.cuda()
            target = target.type(torch.int64).cuda()
            logit, Y_prob, Y_hat = model(input)
            target_wsi.append(target.item())
            # logits = torch.cat((logits, Y_prob.detach().cpu()), dim=0)
            instance_preds[img_id[0]] = Y_prob.detach().cpu().numpy()
    # score, pred = torch.max(logits, dim=1)
    torch.cuda.empty_cache()
    return instance_preds


def train_epoch(epoch, model, optimizer, trainloader, criterion, measure=1, verbose=1):
    model.train()

    logits = torch.Tensor()
    target_wsi_all = []
    attns = {}
    loss_nn_all = 0.
    for i_batch, sampled_batch in enumerate(trainloader):
        optimizer.zero_grad()  # zero the gradient buffer
        img_id, X, target = sampled_batch['img_id'], sampled_batch['feat'], sampled_batch['target']

        input = X.cuda()
        target = target.cuda()

        logit, Y_prob, Y_hat, attn = model(input)

        target_wsi_all.append(target.item())
        logits = torch.cat((logits, logit.detach().cpu()), dim=0)
        attns[img_id[0]] = attn
        loss_cls_wsi = criterion(logit, target)
        loss = loss_cls_wsi

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        loss_nn_all += loss.detach().item()

    if measure:
        target_wsi_all = np.asarray(target_wsi_all)
        _, pred = torch.max(logits, dim=1)
        eq = np.equal(target_wsi_all, pred)
        acc = float(eq.sum()) / target_wsi_all.shape[0]

        if verbose > 0:
            print("\nEpoch: {}, loss_all: {:.4f}, WSI ACC: {:.4f}".format(
                epoch, loss_nn_all / len(trainloader.dataset), acc))
        return attns


if __name__ == '__main__':
    batch_size = args.batch_size
    num_epochs = args.epochs
    feat_dir = args.feat_dir
    lr = args.lr

    test_accs = []
    start = time.time()

    for k in range(args.k):
        model_save_dir = args.results_dir
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        train_dataset, val_dataset, test_dataset = return_splits(from_id=False,
                                                                 csv_path='{}/fold{}.csv'.format(args.split_dir, k))

        train_dset = MyDataset(feat_dir=args.feat_dir, train=True, split=train_dataset)
        trainloader = DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=0)

        val_dset = MyDataset(feat_dir=args.feat_dir, train=False, split=val_dataset)
        valloader = DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=0)

        test_dset = MyDataset(feat_dir=args.feat_dir, train=False, split=test_dataset)
        testloader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=0)


        model = BAL_P(n_classes=2, input_dim=args.encoding_size, k_sample=10, subtyping=False).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        best_acc = 0

        model_path = os.path.join(model_save_dir, f't{args.round}_primary.pth')
        early_stopping = EarlyStopping(model_path=model_path, patience=10, verbose=True)

        ## training the MIL classifier
        for epoch in range(args.epochs):
            train_attns = train_epoch(epoch, model, optimizer, trainloader, criterion)
            valid_loss, val_auc, val_attns = prediction(model, valloader, criterion)

            early_stopping(epoch, valid_loss, best_acc, model)
            if early_stopping.early_stop:
                print('Early Stopping')
                break
            print('\r')


        trainloader = DataLoader(train_dset, batch_size=1, shuffle=False, num_workers=4)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

        _, test_acc, test_attns = prediction(model, testloader, criterion, testing=True)

        _, _, train_attns = prediction(model, trainloader, criterion, testing=True)
        _, _, val_attns = prediction(model, valloader, criterion, testing=True)

        if args.round == 0:
            test_preds, train_preds, val_preds = None, None, None
        else:
            model = BAL_A(n_classes=2, input_dim=args.encoding_size).cuda()
            model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
            test_preds = patch_prediction(model, testloader, criterion, testing=True)
            train_preds = patch_prediction(model, trainloader, criterion, testing=True)
            val_preds = patch_prediction(model, valloader, criterion, testing=True)

        obj = {
            'train_attns': train_attns,
            'train_preds': train_preds,
            'val_attns': val_attns,
            'val_preds': val_preds,
            'test_attns': test_attns,
            'test_preds': test_preds
        }
        with open(os.path.join(model_save_dir, f't{args.round+1}_primary_attn.pkl'), 'wb') as f:
            pickle.dump(obj, f)

        print('Fold: %d => ACC: %f' % (k, test_acc))
        test_accs.append(test_acc)
    end = time.time()
    print('use time: ', end - start)
    print("All Fold Acc: ", test_accs)
    print("Mean Acc: ", np.mean(test_accs))
    print("Std Acc: ", np.std(test_accs))

