#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2022.

import argparse
import logging
from distutils.util import strtobool
from pathlib import Path

from ace_trainer import TrainerACE


def _strtobool(x):
    return bool(strtobool(x))


if __name__ == '__main__':

    # Setup logging levels.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Fast training of a scene coordinate regression network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    """
    ACE Parser
    """
    parser.add_argument('scene', type=Path,
                        help='path to a scene in the dataset folder, e.g. "datasets/Cambridge_KingsCollege"')

    parser.add_argument('output_map_file', type=Path,
                        help='target file for the trained network')

    parser.add_argument('--encoder_path', type=Path, default=Path(__file__).parent / "ace_encoder_pretrained.pt",
                        help='file containing pre-trained encoder weights')

    parser.add_argument('--num_head_blocks', type=int, default=1,
                        help='depth of the regression head, defines the map size')

    parser.add_argument('--learning_rate_min', type=float, default=0.0005,
                        help='lowest learning rate of 1 cycle scheduler')

    parser.add_argument('--learning_rate_max', type=float, default=0.005,
                        help='highest learning rate of 1 cycle scheduler')

    parser.add_argument('--training_buffer_size', type=int, default=8000000,
                        help='number of patches in the training buffer')

    parser.add_argument('--spatial_clusters', type=int, default=64)

    parser.add_argument('--feature_clusters', type=int, default=2)
    parser.add_argument('--stage_len', type=int, default=4,
                        help='number of epochs per training stage')
    parser.add_argument('--use_ema_centers', action='store_true',
                        help='update spatial cluster centers with EMA smoothing')
    parser.add_argument('--ema_beta', type=float, default=0.9,
                        help='EMA decay for spatial cluster center updates')
    parser.add_argument('--use_depth_for_coord', type=_strtobool, default=True,
                        help='use depth maps to generate scene coordinates')
    parser.add_argument('--use_depth_rank_loss', action='store_true',
                        help='enable depth rank loss based on relative depth ordering')
    parser.add_argument('--use_depth_quantile_loss', action='store_true',
                        help='enable depth quantile matching loss')
    parser.add_argument('--use_depth_dist_loss', action='store_true',
                        help='enable depth distribution matching loss')
    parser.add_argument('--depth_rank_weight', type=float, default=1.0,
                        help='weight for depth rank loss')
    parser.add_argument('--depth_quantile_weight', type=float, default=1.0,
                        help='weight for depth quantile loss')
    parser.add_argument('--depth_dist_weight', type=float, default=1.0,
                        help='weight for depth distribution loss')
    parser.add_argument('--depth_rank_pairs_per_image', type=int, default=256,
                        help='number of depth rank pairs per image')
    parser.add_argument('--min_samples_per_image', type=int, default=32,
                        help='minimum samples per image to compute depth distribution losses')
    parser.add_argument('--depth_quantiles', type=str, default='0.1,0.5,0.9',
                        help='comma-separated quantiles for depth quantile loss')
    parser.add_argument('--depth_normalize', type=str, default='zscore',
                        choices=['none', 'zscore', 'minmax'],
                        help='normalization for depth values before loss computation')
    parser.add_argument('--use_depth_bucket_sampling', action='store_true',
                        help='enable depth bucket sampling with rotation cache')
    parser.add_argument('--depth_bucket_bins', type=int, default=4,
                        help='number of depth buckets for balanced sampling')
    parser.add_argument('--depth_bucket_ratio', type=str, default='',
                        help='comma-separated ratios per depth bucket')
    parser.add_argument('--depth_bucket_ratio_start', type=str, default='',
                        help='starting ratios for depth bucket annealing')
    parser.add_argument('--depth_bucket_ratio_end', type=str, default='',
                        help='ending ratios for depth bucket annealing')
    parser.add_argument('--depth_bucket_ratio_anneal_epochs', type=int, default=0,
                        help='epochs to anneal depth bucket ratios')
    parser.add_argument('--depth_bucket_with_replacement', type=_strtobool, default=True,
                        help='sample with replacement when a depth bucket is exhausted')
    parser.add_argument('--depth_bucket_shuffle_each_epoch', type=_strtobool, default=False,
                        help='shuffle depth bucket indices each epoch')
    parser.add_argument('--depth_bucket_separate_invalid', type=_strtobool, default=False,
                        help='separate invalid depth values into their own bucket')
    parser.add_argument('--depth_bucket_invalid_ratio', type=float, default=0.1,
                        help='ratio for invalid depth bucket when separated')
    parser.add_argument('--use_depth_bucket_quality_weighting', type=_strtobool, default=False,
                        help='weight depth bucket sampling by per-sample quality')
    parser.add_argument('--depth_bucket_quality_momentum', type=float, default=0.9,
                        help='EMA momentum for depth bucket quality updates')
    parser.add_argument('--log_depth_bucket_stats', type=_strtobool, default=False,
                        help='log depth bucket sampling statistics each epoch')
    parser.add_argument('--use_semantic_dp_proto', type=_strtobool, default=False,
                        help='enable semantic DP prototype labels for feature supervision')
    parser.add_argument('--dp_max_proto_per_class', type=int, default=16,
                        help='maximum number of DP prototypes per semantic class')
    parser.add_argument('--dp_tau_mode', type=str, default='fixed', choices=['fixed', 'noise_model'],
                        help='threshold mode for opening new semantic prototypes')
    parser.add_argument('--dp_tau', type=float, default=2.0,
                        help='base threshold for opening new semantic prototypes')
    parser.add_argument('--dp_tau_scale', type=float, default=0.02,
                        help='depth-dependent scale for noise-model threshold')
    parser.add_argument('--dp_proto_ema_beta', type=float, default=0.9,
                        help='EMA update factor for prototype parameter updates')
    parser.add_argument('--dp_default_depth', type=float, default=10.0,
                        help='default pseudo-depth used when depth prior is unavailable')
    parser.add_argument('--dp_use_depth_prior', type=_strtobool, default=True,
                        help='use observed depth as pseudo-point prior for DP prototype updates')
    parser.add_argument('--dp_update_every_epochs', type=int, default=1,
                        help='print/update DP prototype status every N epochs')
    parser.add_argument('--use_depth_weighted_repro', action='store_true',
                        help='enable depth-weighted reprojection error in early epochs')
    parser.add_argument('--depth_weighted_epochs', type=int, default=4,
                        help='number of epochs to apply depth-weighted reprojection')
    parser.add_argument('--depth_weight_source', type=str, default='pred',
                        choices=['pred', 'gt'],
                        help='depth source for reprojection weighting')
    parser.add_argument('--depth_weight_type', type=str, default='inv',
                        choices=['inv', 'log'],
                        help='depth weighting function')
    parser.add_argument('--depth_weight_scale', type=float, default=1.0,
                        help='scale factor for depth weighting')

    parser.add_argument('--samples_per_image', type=int, default=1024,
                        help='number of patches drawn from each image when creating the buffer')

    parser.add_argument('--batch_size', type=int, default=5120,
                        help='number of patches for each parameter update (has to be a multiple of 512)')

    parser.add_argument('--epochs', type=int, default=16,
                        help='number of runs through the training buffer')

    parser.add_argument('--repro_loss_hard_clamp', type=int, default=1000,
                        help='hard clamping threshold for the reprojection losses')

    parser.add_argument('--repro_loss_soft_clamp', type=int, default=50,
                        help='soft clamping threshold for the reprojection losses')

    parser.add_argument('--repro_loss_soft_clamp_min', type=int, default=1,
                        help='minimum value of the soft clamping threshold when using a schedule')

    parser.add_argument('--use_half', type=_strtobool, default=True,
                        help='train with half precision')

    parser.add_argument('--use_homogeneous', type=_strtobool, default=True,
                        help='train with half precision')

    parser.add_argument('--use_aug', type=_strtobool, default=True,
                        help='Use any augmentation.')

    parser.add_argument('--aug_rotation', type=int, default=15,
                        help='max inplane rotation angle')

    parser.add_argument('--aug_scale', type=float, default=1.5,
                        help='max scale factor')

    parser.add_argument('--image_resolution', type=int, default=480,
                        help='base image resolution')

    parser.add_argument('--repro_loss_type', type=str, default="dyntanh",
                        choices=["l1", "l1+sqrt", "l1+log", "tanh", "dyntanh"],
                        help='Loss function on the reprojection error. Dyn varies the soft clamping threshold')

    parser.add_argument('--repro_loss_schedule', type=str, default="circle", choices=['circle', 'linear'],
                        help='How to decrease the softclamp threshold during training, circle is slower first')

    parser.add_argument('--depth_min', type=float, default=0.1,
                        help='enforce minimum depth of network predictions')

    parser.add_argument('--depth_target', type=float, default=10,
                        help='default depth to regularize training')

    parser.add_argument('--depth_max', type=float, default=1000,
                        help='enforce maximum depth of network predictions')

    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')

    parser.add_argument('--output_dir', default='./output/', type=str, help="path where to save the output")


    parser.add_argument("--output_name", default='')

    parser.add_argument("--iters_epoch", default=100, type=int)

    options = parser.parse_args()

    trainer = TrainerACE(options)
    trainer.train()
