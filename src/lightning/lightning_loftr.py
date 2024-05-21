import math
from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path
from torchvision import transforms
import torch
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt
import torch.nn.functional as F
from flow import convert_mapping_to_flow
from src.loftr import LoFTR
from src.loftr.utils.supervision import compute_supervision_coarse, compute_supervision_fine
from src.losses.loftr_loss import LoFTRLoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics
)
from src.utils.plotting import make_matching_figures
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler
from PIL import Image

class PL_LoFTR(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.loftr_cfg = lower_config(_config['loftr'])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        # Matcher: LoFTR
        self.matcher = LoFTR(config=_config['loftr']).eval()
        self.loss = LoFTRLoss(_config)

        # Pretrained weights
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            self.matcher.load_state_dict(state_dict, strict=False)
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")
        
        # Testing
        self.dump_dir = dump_dir

    ####estimate on ETH3D######
    def pre_process_data(self, source_img, target_img):
        """
        Resizes images so that their size is dividable by 64, then scale values to [0, 1].
        Args:
            source_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
            target_img:  torch tensor, bx3xHxW in range [0, 255], not normalized yet

        Returns:
            source_img, target_img, normalized to [0, 1] and put to BGR (according to original PWCNet)
            ratio_x, ratio_y: ratio from original sizes to size dividable by 64.
        """
        b, _, h_scale, w_scale = target_img.shape
        int_preprocessed_width = int(math.floor(math.ceil(w_scale / 64.0) * 64.0))
        int_preprocessed_height = int(math.floor(math.ceil(h_scale / 64.0) * 64.0))

        '''
        source_img = torch.nn.functional.interpolate(input=source_img.float().to(device),
                                                     size=(int_preprocessed_height, int_preprocessed_width),
                                                     mode='area').byte()
        target_img = torch.nn.functional.interpolate(input=target_img.float().to(device),
                                                     size=(int_preprocessed_height, int_preprocessed_width),
                                                     mode='area').byte()
        source_img = source_img.float().div(255.0)
        target_img = target_img.float().div(255.0)
        '''
        # this gives slightly better values
        source_img_copy = torch.zeros((b, 3, int_preprocessed_height, int_preprocessed_width))
        target_img_copy = torch.zeros((b, 3, int_preprocessed_height, int_preprocessed_width))
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((int_preprocessed_height, int_preprocessed_width),
                                                          interpolation=2),
                                        transforms.ToTensor()])
        # only /255 the tensor
        for i in range(source_img.shape[0]):
            source_img_copy[i] = transform(source_img[i].byte())
            target_img_copy[i] = transform(target_img[i].byte())

        source_img = source_img_copy
        target_img = target_img_copy

        ratio_x = float(w_scale) / float(int_preprocessed_width)
        ratio_y = float(h_scale) / float(int_preprocessed_height)

        # convert to BGR
        return source_img[:, [2, 1, 0]].to(self.device), target_img[:, [2, 1, 0]].to(self.device), ratio_x, ratio_y
    def estimate_flow(self, source_img, target_img, output_shape=None, scaling=1.0, mode='channel_first',
                      return_corr=False, *args, **kwargs):
        """
        Estimates the flow field relating the target to the source image. Returned flow has output_shape if provided,
        otherwise the same dimension than the target image. If scaling is provided, the output shape is the
        target image dimension multiplied by this scaling factor.
        Args:
            source_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
            target_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
            output_shape: int or list of int, or None, output shape of the returned flow field
            scaling: float, scaling factor applied to target image shape, to obtain the outputted flow field dimensions
                     if output_shape is None
            mode: if channel_first, flow has shape b, 2, H, W. Else shape is b, H, W, 2

        Returns:
            flow_est: estimated flow field relating the target to the reference image,resized and scaled to output_shape
                      (can be defined by scaling parameter)
        """

        b, _, h_scale, w_scale = target_img.shape
        # define output_shape
        if output_shape is None and scaling != 1.0:
            output_shape = (int(h_scale * scaling), int(w_scale * scaling))
        elif output_shape is None:
            output_shape = (h_scale, w_scale)

        source_img, target_img, ratio_x, ratio_y = self.pre_process_data(source_img, target_img)
        source_img, target_img = source_img.squeeze(), target_img.squeeze()
        transform = transforms.Grayscale()
        source_img, target_img = transform(source_img), transform(target_img)
        source_img, target_img = source_img.unsqueeze(0), target_img.unsqueeze(0)
        batch = {'image0': source_img, 'image1': target_img}
        self.matcher(batch)
        correlation_from_t_to_s = batch['conf_matrix']
        # h_, w_ = correlation_from_t_to_s.shape[-2:]
        # correlation_from_t_to_s = correlation_from_t_to_s.view(b, -1, h_, w_)
        #
        # if self.inference_strategy == 'argmax':
        #     # like in original work
        #     flow_est = correlation_to_flow_w_argmax(correlation_from_t_to_s, output_shape=output_shape,
        #                                             do_softmax=True)
        x_source, y_source, x_target, y_target = batch['mkpts0_f'][:,0], batch['mkpts0_f'][:,1], batch['mkpts1_f'][:,0], batch['mkpts1_f'][:,1]
        # x_source dimension is B x H*W
        H, W = target_img.shape[-2], target_img.shape[-1]
        mapping_est = torch.cat((x_source.unsqueeze(-1), y_source.unsqueeze(-1)), dim=-1).view(b, H, W, 2).permute(0, 3,
                                                                                                                   1, 2)
        # score = score.view(b, H, W)

        # b, 2, H, W
        flow_est = convert_mapping_to_flow(mapping_est)

        if output_shape is not None and (H != output_shape[0] or W != output_shape[1]):
            flow_est = F.interpolate(flow_est, output_shape, mode='bilinear', align_corners=False)
            flow_est[:, 0] *= float(output_shape[1]) / float(W)
            flow_est[:, 1] *= float(output_shape[0]) / float(H)
        if mode != 'channel_first':
            flow_est = flow_est.permute(0, 2, 3, 1)

        if return_corr:
            correlation_from_t_to_s = torch.nn.functional.softmax(correlation_from_t_to_s.view(b, -1, h_, w_), dim=1)
            return flow_est, correlation_from_t_to_s
        return flow_est


    ####################################
        
    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]
    
    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx,
            optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + \
                    (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * \
                    abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
    
    def _trainval_inference(self, batch):
        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)
        
        with self.profiler.profile("LoFTR"):
            self.matcher(batch)
        
        with self.profiler.profile("Compute fine supervision"):
            compute_supervision_fine(batch, self.config)
            
        with self.profiler.profile("Compute losses"):
            self.loss(batch)
    
    def _compute_metrics(self, batch):
        with self.profiler.profile("Copmute metrics"):
            compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
            compute_pose_errors(batch, self.config)  # compute R_errs, t_errs, pose_errs for each pair

            rel_pair_names = list(zip(*batch['pair_names']))
            bs = batch['image0'].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
                'R_errs': batch['R_errs'],
                't_errs': batch['t_errs'],
                'inliers': batch['inliers']}
            ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names
    
    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        
        # logging
        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            # scalars
            for k, v in batch['loss_scalars'].items():
                self.logger.experiment.add_scalar(f'train/{k}', v, self.global_step)

            # net-params
            if self.config.LOFTR.MATCH_COARSE.MATCH_TYPE == 'sinkhorn':
                self.logger.experiment.add_scalar(
                    f'skh_bin_score', self.matcher.coarse_matching.bin_score.clone().detach().cpu().data, self.global_step)

            # figures
            # if self.config.TRAINER.ENABLE_PLOTTING:
            #     compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
            #     figures = make_matching_figures(batch, self.config, self.config.TRAINER.PLOT_MODE)
            #     for k, v in figures.items():
            #         self.logger.experiment.add_figure(f'train_match/{k}', v, self.global_step)

        return {'loss': batch['loss']}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar(
                'train/avg_loss_on_epoch', avg_loss,
                global_step=self.current_epoch)
    
    def validation_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        
        ret_dict, _ = self._compute_metrics(batch)
        
        val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figures = {self.config.TRAINER.PLOT_MODE: []}
        #if batch_idx % val_plot_interval == 0:
            #figures = make_matching_figures(batch, self.config, mode=self.config.TRAINER.PLOT_MODE)

        return {
            **ret_dict,
            'loss_scalars': batch['loss_scalars'],
            'figures': figures,
        }
        
    def validation_epoch_end(self, outputs):
        # handle multiple validation sets
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        multi_val_metrics = defaultdict(list)
        
        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            if not self.trainer.resume_from_checkpoint and self.trainer.running_sanity_check:
                cur_epoch = -1

            # 1. loss_scalars: dict of list, on cpu
            _loss_scalars = [o['loss_scalars'] for o in outputs]
            loss_scalars = {k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) for k in _loss_scalars[0]}

            # 2. val metrics: dict of list, numpy
            _metrics = [o['metrics'] for o in outputs]
            metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
            # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0 
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
            for thr in [5, 10, 20]:
                multi_val_metrics[f'auc@{thr}'].append(val_metrics_4tb[f'auc@{thr}'])
            
            # 3. figures
            # _figures = [o['figures'] for o in outputs]
            # figures = {k: flattenList(gather(flattenList([_me[k] for _me in _figures]))) for k in _figures[0]}

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.logger.experiment.add_scalar(f'val_{valset_idx}/avg_{k}', mean_v, global_step=cur_epoch)

                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch)
                
                # for k, v in figures.items():
                #     if self.trainer.global_rank == 0:
                #         for plot_idx, fig in enumerate(v):
                #             self.logger.experiment.add_figure(
                #                 f'val_match_{valset_idx}/{k}/pair-{plot_idx}', fig, cur_epoch, close=True)
            plt.close('all')

        for thr in [5, 10, 20]:
            # log on all ranks for ModelCheckpoint callback to work properly
            self.log(f'auc@{thr}', torch.tensor(np.mean(multi_val_metrics[f'auc@{thr}'])))  # ckpt monitors on this

    def test_step(self, batch, batch_idx):
        with self.profiler.profile("LoFTR"):
            self.matcher(batch)

        ret_dict, rel_pair_names = self._compute_metrics(batch)
        # figures = make_matching_figures(batch, self.config, path=f"D:\\LoFTR-master-scalenet\\img\\{batch_idx}", mode='evaluation')
        with self.profiler.profile("dump_results"):
            if self.dump_dir is not None:
                # dump results for further analysis
                keys_to_save = {'mkpts0_f', 'mkpts1_f', 'mconf', 'epi_errs'}
                pair_names = list(zip(*batch['pair_names']))
                bs = batch['image0'].shape[0]
                dumps = []
                for b_id in range(bs):
                    item = {}
                    mask = batch['m_bids'] == b_id
                    item['pair_names'] = pair_names[b_id]
                    item['identifier'] = '#'.join(rel_pair_names[b_id])
                    for key in keys_to_save:
                        item[key] = batch[key][mask].cpu().numpy()
                    for key in ['R_errs', 't_errs', 'inliers']:
                        item[key] = batch[key][b_id]
                    dumps.append(item)
                ret_dict['dumps'] = dumps

        return ret_dict

    def test_epoch_end(self, outputs):
        # metrics: dict of list, numpy
        _metrics = [o['metrics'] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

        # [{key: [{...}, *#bs]}, *#batch]
        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o['dumps'] for o in outputs])  # [{...}, #bs*#batch]
            dumps = flattenList(gather(_dumps))  # [{...}, #proc*#bs*#batch]
            logger.info(f'Prediction and evaluation results will be saved to: {self.dump_dir}')

        if self.trainer.global_rank == 0:
            print(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
            logger.info('\n' + pprint.pformat(val_metrics_4tb))
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / 'LoFTR_pred_eval', dumps)
