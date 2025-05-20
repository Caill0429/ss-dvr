#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2022.

import argparse
import logging
import math
import os.path
import time
from distutils.util import strtobool
from pathlib import Path

import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

import dsacstar
from ace_network import Regressor
from dataset import CamLocDataset

import ace_vis_util as vutil
from ace_visualizer import ACEVisualizer

_logger = logging.getLogger(__name__)

sys.path.insert(0, './pnpransac')
from pnpransac import pnpransac
from ace_util import calculate_euclidean

def _strtobool(x):
    return bool(strtobool(x))


if __name__ == '__main__':
    # Setup logging.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Test a trained network on a specific scene.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('scene', type=Path,
                        help='path to a scene in the dataset folder, e.g. "datasets/Cambridge_KingsCollege"')

    parser.add_argument('network', type=Path, help='path to a network trained for the scene (just the head weights)')

    parser.add_argument('--encoder_path', type=Path, default=Path(__file__).parent / "ace_encoder_pretrained.pt",
                        help='file containing pre-trained encoder weights')

    parser.add_argument('--session', '-sid', default='',
                        help='custom session name appended to output files, '
                             'useful to separate different runs of a script')

    parser.add_argument('--image_resolution', type=int, default=480, help='base image resolution')

    # ACE is RGB-only, no need for this param.
    # parser.add_argument('--mode', '-m', type=int, default=1, choices=[1, 2], help='test mode: 1 = RGB, 2 = RGB-D')

    # DSACStar RANSAC parameters. ACE Keeps them at default.
    parser.add_argument('--hypotheses', '-hyps', type=int, default=64,
                        help='number of hypotheses, i.e. number of RANSAC iterations')

    parser.add_argument('--threshold', '-t', type=float, default=10,
                        help='inlier threshold in pixels (RGB) or centimeters (RGB-D)')

    parser.add_argument('--inlieralpha', '-ia', type=float, default=100,
                        help='alpha parameter of the soft inlier count; controls the softness of the '
                             'hypotheses score distribution; lower means softer')

    parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100,
                        help='maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking '
                             'pose consistency towards all measurements; error is clamped to this value for stability')

    # Params for the visualization. If enabled, it will slow down relocalisation considerably. But you get a nice video :)
    parser.add_argument('--render_visualization', type=_strtobool, default=False,
                        help='create a video of the mapping process')

    parser.add_argument('--render_target_path', type=Path, default='renderings',
                        help='target folder for renderings, visualizer will create a subfolder with the map name')

    parser.add_argument('--render_flipped_portrait', type=_strtobool, default=False,
                        help='flag for wayspots dataset where images are sideways portrait')

    parser.add_argument('--render_sparse_queries', type=_strtobool, default=False,
                        help='set to true if your queries are not a smooth video')

    parser.add_argument('--render_pose_error_threshold', type=int, default=20,
                        help='pose error threshold for the visualisation in cm/deg')

    parser.add_argument('--render_map_depth_filter', type=int, default=10,
                        help='to clean up the ACE point cloud remove points too far away')

    parser.add_argument('--render_camera_z_offset', type=int, default=4,
                        help='zoom out of the scene by moving render camera backwards, in meters')

    parser.add_argument('--render_frame_skip', type=int, default=1,
                        help='skip every xth frame for long and dense query sequences')

    parser.add_argument('--spatial_clusters', type=int, default=64)

    parser.add_argument('--feature_clusters', type=int, default=2)

    parser.add_argument('--conf_threshold', type=float, default=0.51)

    opt = parser.parse_args()

    device = torch.device("cuda")
    num_workers = 6

    scene_path = Path(opt.scene)
    head_network_path = Path(opt.network)
    encoder_path = Path(opt.encoder_path)
    session = opt.session

    # Setup dataset.
    testset = CamLocDataset(
        scene_path / "test",
        mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
        image_height=opt.image_resolution,
    )
    _logger.info(f'Test images found: {len(testset)}')

    # Setup dataloader. Batch size 1 by default.
    testset_loader = DataLoader(testset, shuffle=False, num_workers=6)

    # Load network weights.
    encoder_state_dict = torch.load(encoder_path, map_location="cpu")
    _logger.info(f"Loaded encoder from: {encoder_path}")
    head_state_dict = torch.load(head_network_path, map_location="cpu")
    _logger.info(f"Loaded head weights from: {head_network_path}")

    # Create regressor.
    network = Regressor.create_from_split_state_dict(encoder_state_dict, head_state_dict, opt.feature_clusters, opt.spatial_clusters)

    # Setup for evaluation.
    network = network.to(device)
    network.eval()

    # Save the outputs in the same folder as the network being evaluated.
    output_dir = head_network_path.parent
    scene_name = scene_path.name
    # This will contain aggregate scene stats (median translation/rotation errors, and avg processing time per frame).
    # test_log_file = output_dir / f'test_{scene_name}_{opt.session}.txt'
    # _logger.info(f"Saving test aggregate statistics to: {test_log_file}")
    # This will contain each frame's pose (stored as quaternion + translation) and errors.
    # pose_log_file = output_dir / f'poses_{scene_name}_{opt.session}.txt'
    # _logger.info(f"Saving per-frame poses and errors to: {pose_log_file}")

    # Setup output files.
    # test_log = open(test_log_file, 'w', 1)
    # pose_log = open(pose_log_file, 'w', 1)

    # Metrics of interest.
    avg_batch_time = 0
    num_batches = 0

    # Keep track of rotation and translation errors for calculation of the median error.
    rErrs = []
    tErrs = []
    dErrs = []

    # Percentage of frames predicted within certain thresholds from their GT pose.
    pct10_5 = 0
    pct5 = 0
    pct2 = 0
    pct1 = 0

    # Generate video of training process
    if opt.render_visualization:
        # infer rendering folder from map file name
        target_path = vutil.get_rendering_target_path(
            opt.render_target_path,
            opt.network)
        ace_visualizer = ACEVisualizer(target_path,
                                       opt.render_flipped_portrait,
                                       opt.render_map_depth_filter,
                                       reloc_vis_error_threshold=opt.render_pose_error_threshold)

        # we need to pass the training set in case the visualiser has to regenerate the map point cloud
        trainset = CamLocDataset(
            scene_path / "train",
            mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
            image_height=opt.image_resolution,
        )

        # Setup dataloader. Batch size 1 by default.
        trainset_loader = DataLoader(trainset, shuffle=False, num_workers=6)

        ace_visualizer.setup_reloc_visualisation(
            frame_count=len(testset),
            data_loader=trainset_loader,
            network=network,
            camera_z_offset=opt.render_camera_z_offset,
            reloc_frame_skip=opt.render_frame_skip)
    else:
        ace_visualizer = None

    # Testing loop.
    testing_start_time = time.time()
    count = 0
    with torch.no_grad():
        for image_B1HW, image_mask_B1HW, coord, gt_pose_B44, _, intrinsics_B33, _, filenames in testset_loader:
            batch_start_time = time.time()
            batch_size = image_B1HW.shape[0]

            image_B1HW = image_B1HW.to(device, non_blocking=True)

            # Predict scene coordinates.
            with autocast(enabled=True):
                scene_coordinates_B4HW, _, _, clusters_int = network(image_B1HW)

            img = clusters_int[0][0].cpu().detach().numpy()
            img = img / img.max()
            img_name = str(filenames[0]).split('/')[-1]
            img_scene = str(opt.scene).split('/')[-1]
            img_dir = os.path.join('clusters', img_scene)
            img_name = os.path.join(img_dir, img_name)

            # if not os.path.exists(img_dir):
            #     os.makedirs(img_dir)
            # plt.imsave(img_name, img)

            scene_coordinates_B3HW = scene_coordinates_B4HW[:, 0:3, :, :]
            scene_coordinates_HW = scene_coordinates_B4HW[0, 3, :, :]
            threshold = torch.quantile(scene_coordinates_HW, opt.conf_threshold)
            conf_mask = (scene_coordinates_HW > threshold).flatten().cpu().detach().numpy()

            conf_numpy = scene_coordinates_HW.cpu().detach().numpy()
            conf_numpy = (conf_numpy - conf_numpy.min()) / (conf_numpy.max() - conf_numpy.min())

            B, _, H, W = scene_coordinates_B3HW.size()
            coord = coord[0, 4::8, 4::8, :].cpu()
            # H, W, _ = coord.size()
            x = np.linspace(0, 1999, 2000)
            y = np.linspace(0, 1999, 2000)
            xx, yy = np.meshgrid(x, y)
            pcoord = np.concatenate((np.expand_dims(xx, axis=2),
                                     np.expand_dims(yy, axis=2)), axis=2)
            pcoord = pcoord[4::8, 4::8]
            pcoord = pcoord[:H, :W]

            pcoord = np.ascontiguousarray(pcoord)

            pred_xyz = scene_coordinates_B3HW[0, 0:3, :, :].permute(1, 2, 0).cpu()
            # calculate_euclidean(pred_xyz, coord, image_mask_B1HW, scene_coordinates_HW)

            pred_xyz = np.ascontiguousarray(pred_xyz)
            truth_xyz = np.ascontiguousarray(coord)

            param1 = np.reshape(pcoord, (-1, 2)).astype(np.float64)
            param2 = np.reshape(pred_xyz, (-1, 3)).astype(np.float64)
            param3 = np.reshape(truth_xyz, (-1, 3)).astype(np.float64)

            param1_pnp = param1[conf_mask, :]
            param2_pnp = param2[conf_mask, :]
            param3_pnp = param3[conf_mask, :]

            intrinsics_color = intrinsics_B33[0]
            pose_solver = pnpransac(intrinsics_color[0, 0], intrinsics_color[1, 1],
                                    intrinsics_color[0, 2], intrinsics_color[1, 2])

            rot, transl = pose_solver.RANSAC_loop(param1_pnp, param2_pnp, 256)

            mask = (image_mask_B1HW == 1)
            mask = mask[0, 0, 4::8, 4::8].flatten()

            param1 = param1[mask, :]
            param2 = param2[mask, :]
            param3 = param3[mask, :]
            dist = torch.norm(torch.tensor(param2) - torch.tensor(param3), dim=1).mean().item()

            pose_gt = gt_pose_B44[0]
            pose_est = np.eye(4)
            pose_est[0:3, 0:3] = cv2.Rodrigues(rot)[0].T
            pose_est[0:3, 3] = -np.dot(pose_est[0:3, 0:3], transl)

            def get_pose_err(pose_gt, pose_est):
                transl_err = np.linalg.norm(pose_gt[0:3, 3] - pose_est[0:3, 3])
                rot_err = pose_est[0:3, 0:3].T.dot(pose_gt[0:3, 0:3])
                rot_err = cv2.Rodrigues(rot_err)[0]
                rot_err = np.reshape(rot_err, (1, 3))
                rot_err = np.reshape(np.linalg.norm(rot_err, axis=1), -1) / np.pi * 180.
                return transl_err, rot_err[0], (pose_gt[0:3, 3] - pose_est[0:3, 3])

            transl_err, rot_err, cam_xyz = get_pose_err(pose_gt, pose_est)

            _logger.info(f"{str(filenames)} Rotation Error: {rot_err:.3f}deg, Translation Error: {transl_err * 100:.3f}cm, Dist: {dist:3f}")

            count += 1

            rErrs.append(rot_err)
            tErrs.append(transl_err * 100)
            dErrs.append(dist)

            # Check various thresholds.
            if rot_err < 5 and transl_err < 0.1:  # 10cm/5deg
                pct10_5 += 1
            if rot_err < 5 and transl_err < 0.05:  # 5cm/5deg
                pct5 += 1
            if rot_err < 2 and transl_err < 0.02:  # 2cm/2deg
                pct2 += 1
            if rot_err < 1 and transl_err < 0.01:  # 1cm/1deg
                pct1 += 1

            avg_batch_time += time.time() - batch_start_time
            num_batches += 1

    total_frames = len(rErrs)
    assert total_frames == len(testset)

    # Compute median errors.
    tErrs.sort()
    rErrs.sort()
    dErrs.sort()
    median_idx = total_frames // 2
    median_rErr = rErrs[median_idx]
    median_tErr = tErrs[median_idx]
    median_dErr = dErrs[median_idx]

    # Compute average time.
    avg_time = avg_batch_time / num_batches

    # Compute final metrics.
    pct10_5 = pct10_5 / total_frames * 100
    pct5 = pct5 / total_frames * 100
    pct2 = pct2 / total_frames * 100
    pct1 = pct1 / total_frames * 100

    logger = open('output/_test_result.txt', 'a', 1)
    logger.writelines(f"threshold: {opt.conf_threshold}")
    logger.writelines(f"{head_network_path}\n")
    logger.writelines(f'\t10cm/5deg: {pct10_5:.1f}%\n')
    logger.writelines(f'\t5cm/5deg: {pct5:.1f}%\n')
    logger.writelines(f'\t2cm/2deg: {pct2:.1f}%\n')
    logger.writelines(f'\t1cm/1deg: {pct1:.1f}%\n')
    logger.writelines(f"Median Error: {median_rErr:.5f}deg, {median_tErr:.5f}cm, {median_dErr:.5f}m\n")
    logger.writelines(f"Avg. processing time: {avg_time * 1000:4.1f}ms\n\n")

    _logger.info("===================================================")
    _logger.info("Test complete.")
    _logger.info('Accuracy:')
    _logger.info(f'\t10cm/5deg: {pct10_5:.1f}%')
    _logger.info(f'\t5cm/5deg: {pct5:.1f}%')
    _logger.info(f'\t2cm/2deg: {pct2:.1f}%')
    _logger.info(f'\t1cm/1deg: {pct1:.1f}%')

    _logger.info(f"Median Error: {median_rErr:.5f}deg, {median_tErr:.5f}cm")
    _logger.info(f"Avg. processing time: {avg_time * 1000:4.1f}ms")

    # Write to the test log file as well.
    # test_log.write(f"{median_rErr} {median_tErr} {avg_time}\n")
    #
    # test_log.close()
    # pose_log.close()
