from models.model_utils import *
from models.resnet_custom import resnet50_baseline


class BAL_P(nn.Module):
    def __init__(self, input_dim=512, n_classes=6, subtyping=False):
        super(BAL_P, self).__init__()
        self.n_classes = n_classes
        fc_size = [input_dim, 1024, 256]
        # self.path_attention_head = Attn_Net_Gated(L=fc_size[1], D=fc_size[2], dropout=0.2, n_classes=1)
        self.path_attention_head = Attn_Net(L = fc_size[1], D = fc_size[2], dropout=True, n_classes=1)
        self.subtyping = subtyping
        self.classifiers = nn.Linear(fc_size[1], n_classes)

    def forward(self, wsi_h, ):
        B = wsi_h.shape[0]  # batch size
        wsi_N = wsi_h.shape[1]  # number of patch
        wsi_trans = wsi_h.squeeze(0)

        ## Attention Pooling
        A_path, _ = self.path_attention_head(wsi_trans)
        ori_A_path = A_path.view(1, -1)
        A_path = F.softmax(ori_A_path, dim=1)

        M = torch.mm(A_path, wsi_trans)  ## all instance

        # attn = A_path.reshape(1, A_path.size(-1)).detach().cpu().numpy()
        vis_attn = ori_A_path.reshape(1, ori_A_path.size(-1)).detach().cpu().numpy()
        attn = A_path.detach().cpu().numpy()

        # ---->predict (cox head)
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        return logits, Y_prob, Y_hat, attn


class BAL_A(nn.Module):
    def __init__(self, input_dim=1024, n_classes=2):
        super(BAL_A, self).__init__()
        self.fc = nn.Linear(1024, 2)

    def forward(self, patch_h):
        patch_h = patch_h.squeeze(0)
        logits = self.fc(patch_h)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        return logits, Y_prob, Y_hat


