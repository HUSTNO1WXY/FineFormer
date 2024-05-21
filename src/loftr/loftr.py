import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from einops.einops import rearrange
from matplotlib import pyplot as plt
from torchvision import transforms
from .backbone import build_backbone, ResNetFPN_8_2
from .loftr_module.fine import LocalFeatureTransformer
from .utils.position_encoding import PositionEncodingSine
from .loftr_module import FinePreprocess, LocalFeatureTransformer_UNet
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching


class LoFTR(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = build_backbone(config)
        self.pos_encoding = PositionEncodingSine(config['coarse']['d_model'], pre_scaling=[[1024, 1024], [1024, 1024]])
        self.loftr_coarse = LocalFeatureTransformer_UNet(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_preprocess = FinePreprocess(config)
        self.loftr_fine = LocalFeatureTransformer(config["fine"])
        self.fine_matching = FineMatching()

    def forward(self, data, online_resize=False):
        """
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        if online_resize:
            assert data['image0'].shape[0] == 1 and data['image1'].shape[1] == 1
            self.resize_input(data, self.config['coarse']['train_res'])
        else:
            data['pos_scale0'], data['pos_scale1'] = None, None

            # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        with torch.no_grad():
            if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
                feats_c, feats_f = self.backbone(
                    torch.cat([data['image0'], data['image1']], dim=0))
                (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(
                    data['bs']), feats_f.split(data['bs'])
            else:  # handle different input shapes
                (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(
                    data['image0']), self.backbone(data['image1'])

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        # 2. coarse-level loftr module
        [feat_c0, pos_encoding0], [feat_c1, pos_encoding1] = self.pos_encoding(feat_c0,data['pos_scale0']), self.pos_encoding(feat_c1,data['pos_scale1'])

        # ##PCA降维####
        # pos_encoding0, pos_encoding1 = pos_encoding0.reshape(1, 256, 6400).transpose(1,2), pos_encoding1.reshape(1, 256, 6400).transpose(1,2),
        # P0, P1 = pos_encoding0.squeeze(), pos_encoding1.squeeze()
        # P0, P1 = P0.cpu().numpy(), P1.cpu().numpy()
        # P0_pca, P1_pca = pca(P0, 3), pca(P1, 3)
        # P0_pca, P1_pca = torch.from_numpy(P0_pca), torch.from_numpy(P1_pca)
        # P0_pca, P1_pca = P0_pca.cuda(), P1_pca.cuda()
        # P0_pca, P1_pca = P0_pca.view(data['hw0_c'][0], data['hw0_c'][1], -1), P1_pca.view(data['hw1_c'][0],
        #                                                                                   data['hw1_c'][1], -1)
        # P0_pca, P1_pca = P0_pca.cpu().numpy(), P1_pca.cpu().numpy()
        # plt.imshow(P0_pca)
        # plt.savefig('D:\AAAI-supp\\rebuttal-our0')

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)

        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, feat_f0, feat_f1, pos_encoding0, pos_encoding1, mask_c0, mask_c1)


        # # attention map visualization
        # # normalize
        # # feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1] ** .5,
        # #                        [feat_c0, feat_c1])
        # V0, V1 = feat_c0, feat_c1
        # att = torch.einsum("nld,nsd->nls", V0, V1)
        # A0 = torch.mean(att, dim=-2)
        # # A0 = torch.mean(A, dim=-2)
        # # A0 = torch.max(A, dim=-1)[1]
        # A0 = A0.squeeze().view(data['hw0_c'][0], data['hw0_c'][1])
        # A0 = A0.cpu().numpy()
        #
        # img0_pth = "D:\IMG1.jpg"
        # img1_pth = "D:\IMG0.jpg"
        # # img0_raw = cv2.imread(img0_pth)
        # # img1_raw = cv2.imread(img1_pth)
        # # img0_raw = cv2.resize(img0_raw, (
        # # img0_raw.shape[1] // 8 * 8, img0_raw.shape[0] // 8 * 8))  # input size shuold be divisible by 8
        # # img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1] // 8 * 8, img1_raw.shape[0] // 8 * 8))
        #
        # img = Image.open(img1_pth, mode='r')
        # img_h, img_w = img.size[0], img.size[1]
        # plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))
        # img_h, img_w = int(img.size[0] * 1), int(img.size[1] * 1)
        # img = img.resize((img_h, img_w))
        # plt.imshow(img, alpha=1)
        # plt.axis('off')
        #
        # A0 = A0.astype('uint8')
        # mask = cv2.resize(A0, (img_h, img_w))
        # normed_mask = mask / mask.max()
        # normed_mask = (normed_mask * 255).astype('uint8')
        # plt.imshow(normed_mask, alpha=0.6, interpolation='nearest', cmap="jet")
        # plt.savefig('D:\\AAAI-re0-0.jpg')


        #start_time = time.time()
        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)
        # end_time = time.time()
        # print("used time is{}".format(end_time - start_time))

        # 4. fine-level refinement
        # merged by new descriptors\
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)

        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def resize_input(self, data, train_res, df=32):
        h0, w0, h1, w1 = data['image0'].shape[2], data['image0'].shape[3], data['image1'].shape[2], \
                         data['image1'].shape[3]
        data['image0'], data['image1'] = self.resize_df(data['image0'], df), self.resize_df(data['image1'],
                                                                                            df)

        if len(train_res) == 1:
            train_res_h = train_res_w = train_res
        else:
            train_res_h, train_res_w = train_res[0], train_res[1]
        data['pos_scale0'], data['pos_scale1'] = [train_res_h / data['image0'].shape[2],
                                                  train_res_w / data['image0'].shape[3]], \
                                                 [train_res_h / data['image1'].shape[2],
                                                  train_res_w / data['image1'].shape[3]]
        data['online_resize_scale0'], data['online_resize_scale1'] = \
        torch.tensor([w0 / data['image0'].shape[3], h0 / data['image0'].shape[2]])[None].cuda(), \
        torch.tensor([w1 / data['image1'].shape[3], h1 / data['image1'].shape[2]])[None].cuda()

    def resize_df(self, image, df=32):
        h, w = image.shape[2], image.shape[3]
        h_new, w_new = h // df * df, w // df * df
        if h != h_new or w != w_new:
            img_new = transforms.Resize([h_new, w_new]).forward(image)
        else:
            img_new = image
        return img_new

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)


# def pca(data, n_dim):
#     '''
#     pca is O(D^3)
#     :param data: (n_samples, n_features(D))
#     :param n_dim: target dimensions
#     :return: (n_samples, n_dim)
#     '''
#     data = data - np.mean(data, axis = 0, keepdims = True)
#
#     cov = np.dot(data.T, data)
#
#     eig_values, eig_vector = np.linalg.eig(cov)
#     # print(eig_values)
#     indexs_ = np.argsort(-eig_values)[:n_dim]
#     picked_eig_values = eig_values[indexs_]
#     picked_eig_vector = eig_vector[:, indexs_]
#     data_ndim = np.dot(data, picked_eig_vector)
#     return data_ndim