import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.model_utils import *
from torchvision.models import resnet18
from models.resnet_custom import resnet50_baseline


class BAL_P(nn.Module):
    def __init__(self, input_dim=512, n_classes=6, subtyping=False):
        super(BAL_P, self).__init__()
        self.n_classes = n_classes
        fc_size = [input_dim, 1024, 256]
        # self.path_attention_head = Attn_Net_Gated(L=fc_size[1], D=fc_size[2], dropout=0.2, n_classes=n_classes)
        self.path_attention_head = Attn_Net(L = fc_size[1], D = fc_size[2], dropout=True, n_classes=n_classes)
        self.subtyping = subtyping
        # self.classifiers = nn.Linear(fc_size[1], n_classes)
        bag_classifiers = [nn.Linear(fc_size[1], 1) for i in range(n_classes)]  # use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)


    def forward(self, wsi_h):
        B = wsi_h.shape[0]  # batch size
        wsi_N = wsi_h.shape[1]  # number of patch
        wsi_trans = wsi_h.squeeze(0)

        ## Attention Pooling
        A_path, _ = self.path_attention_head(wsi_trans)
        # ori_A_path = A_path.view(1, -1)
        A_path = torch.transpose(A_path, 1, 0)
        A_path = F.softmax(A_path, dim=1)
        M = torch.mm(A_path, wsi_trans)  ## all instance

        # attn = A_path.reshape(1, A_path.size(-1)).detach().cpu().numpy()
        attn = A_path.detach().cpu().numpy()

        logits = torch.empty(1, self.n_classes).float().cuda()
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)

        return logits, Y_prob, Y_hat, attn


class BAL_A(nn.Module):
    def __init__(self, input_dim=512, n_classes=2):
        super(BAL_A, self).__init__()
        # output_node = self.n_classes + 1 if self.n_classes > 2 else 2
        # self.instance_classifiers = nn.Linear(512, output_node)
        self.fc = nn.Linear(1024, n_classes +1)

    def forward(self, patch_h):
        patch_h = patch_h.squeeze(0)
        logits = self.fc(patch_h)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        return logits, Y_prob, Y_hat


if __name__ == "__main__":
    wsi_data = torch.randn((1, 600, 1024)).cuda()
    model = BAL_P(input_dim=1024, n_classes=4).cuda()
    # print(model.eval())
    logits, Y_prob, Y_hat, attn = model(wsi_data)
    print('logits size: ', logits.size())
    print('Y_prob size: ', Y_prob.size())
    print('Y_hat size: ', Y_hat.size())
    print('attn size: ', attn.shape)

