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

    parser.add_argument("--is_train", default=True)

    parser.add_argument("--output_name", default='')

    parser.add_argument("--iters", default=1, type=int)

    parser.add_argument("--iters_epoch", default=100, type=int)

    parser.add_argument("--square", default=1, type=int)

    parser.add_argument("--kmeans_seed", default=0, type=int)

    options = parser.parse_args()

    trainer = TrainerACE(options)
    trainer.train()
