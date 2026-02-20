# Copyright © Niantic, Inc. 2022.

import logging
import random
import time
import cv2
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.nn.functional as F
from ace_util import get_pixel_grid, to_homogeneous
from ace_loss import ReproLoss, EuclideanLoss, CELoss
from ace_network import Regressor
from dataset import CamLocDataset
from tqdm import tqdm
from kmeans import MiniBatchKMeansCUDA
from semantic_dp_prototypes import SemanticDPPrototypeManager

_logger = logging.getLogger(__name__)


def set_seed(seed):
    """
    Seed all sources of randomness.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class TrainerACE:
    def __init__(self, options):
        self.options = options
        self._ensure_depth_options()

        self.device = torch.device('cuda')

        if hasattr(self.options, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = bool(self.options.allow_tf32)
            torch.backends.cudnn.allow_tf32 = bool(self.options.allow_tf32)
        if hasattr(self.options, 'matmul_precision'):
            torch.set_float32_matmul_precision(self.options.matmul_precision)

        # Setup randomness for reproducibility.
        self.base_seed = 2089
        set_seed(self.base_seed)

        # Used to generate batch indices.
        self.batch_generator = torch.Generator()
        self.batch_generator.manual_seed(self.base_seed + 1023)

        # Dataloader generator, used to seed individual workers by the dataloader.
        self.loader_generator = torch.Generator()
        self.loader_generator.manual_seed(self.base_seed + 511)

        # Generator used to sample random features (runs on the GPU).
        self.sampling_generator = torch.Generator(device=self.device)
        self.sampling_generator.manual_seed(self.base_seed + 4095)

        # Generator used to permute the feature indices during each training epoch.
        self.training_generator = torch.Generator()
        self.training_generator.manual_seed(self.base_seed + 8191)

        self.iteration = 0
        self.training_start = None
        self.num_data_loader_workers = max(0, int(getattr(self.options, 'num_workers', 12)))
        self.loader_pin_memory = bool(getattr(self.options, 'loader_pin_memory', True))
        self.loader_prefetch_factor = max(1, int(getattr(self.options, 'loader_prefetch_factor', 2)))

        # Create dataset.
        self.dataset = CamLocDataset(
            root_dir=self.options.scene / "train",
            mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
            use_half=self.options.use_half,
            image_height=self.options.image_resolution,
            augment=self.options.use_aug,
            aug_rotation=self.options.aug_rotation,
            aug_scale_max=self.options.aug_scale,
            aug_scale_min=1 / self.options.aug_scale,
            return_depth=True,
            return_idx=True,
            use_depth_for_coord=self.options.use_depth_for_coord,
        )
        self.use_semantic_labels = self.dataset.semantic_files is not None
        self.use_depth = self.dataset.depth_files is not None

        _logger.info("Loaded training scan from: {} -- {} images, mean: {:.2f} {:.2f} {:.2f}".format(
            self.options.scene,
            len(self.dataset),
            self.dataset.mean_cam_center[0],
            self.dataset.mean_cam_center[1],
            self.dataset.mean_cam_center[2])
        )

        # Create network using the state dict of the pretrained encoder.
        encoder_state_dict = torch.load(self.options.encoder_path, map_location="cpu")

        self.regressor = Regressor.create_from_encoder(
            encoder_state_dict,
            mean=self.dataset.mean_cam_center,
            num_head_blocks=self.options.num_head_blocks,
            use_homogeneous=self.options.use_homogeneous,
            feature_clusters=self.options.feature_clusters,
            spatial_clusters=self.options.spatial_clusters
        )
        _logger.info(f"Loaded pretrained encoder from: {self.options.encoder_path}")

        self.regressor = self.regressor.to(self.device)
        self.regressor.train()

        # Setup optimization parameters.
        self.optimizer = optim.AdamW(self.regressor.parameters(), lr=self.options.learning_rate_min)
        # Setup learning rate scheduler.
        steps_per_epoch = self.options.training_buffer_size // self.options.batch_size
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.options.learning_rate_max,
            epochs=self.options.epochs,
            steps_per_epoch=steps_per_epoch,
            cycle_momentum=False
        )

        # Gradient scaler in case we train with half precision.
        self.scaler = GradScaler(enabled=self.options.use_half)

        # Generate grid of target reprojection pixel positions.
        # pixel_grid_2HW.size() = torch.Size([2, 625, 625])
        pixel_grid_2HW = get_pixel_grid(self.regressor.OUTPUT_SUBSAMPLE)
        self.pixel_grid_2HW = pixel_grid_2HW.to(self.device)

        # Compute total number of iterations.
        self.iterations = self.options.epochs * self.options.training_buffer_size // self.options.batch_size
        self.iterations_output = 100  # print loss every n iterations, and (optionally) write a visualisation frame

        # Setup reprojection loss function.
        self.repro_loss = ReproLoss(
            total_iterations=self.iterations,
            soft_clamp=self.options.repro_loss_soft_clamp,
            soft_clamp_min=self.options.repro_loss_soft_clamp_min,
            type=self.options.repro_loss_type,
            circle_schedule=(self.options.repro_loss_schedule == 'circle')
        )
        self.euclidean_loss = EuclideanLoss(10)
        self.cls_loss = CELoss()
        self.depth_quantiles = [float(q) for q in self.options.depth_quantiles.split(',') if q]

        # Will be filled at the beginning of the training process.
        self.training_buffer = None
        self.conf_alpha = 20
        self.alpha = 0.1
        self.mini_batch_kmeans = MiniBatchKMeansCUDA(n_clusters=self.options.feature_clusters, batch_size=400000, max_iter=200, device='cuda')
        self.depth_bucket_indices = None
        self.depth_bucket_cursors = None
        self.depth_bucket_generator = torch.Generator()
        self.depth_bucket_generator.manual_seed(self.base_seed + 12345)
        self.depth_bucket_last_epoch = None
        self.semantic_dp_manager = None
        if self.options.use_semantic_dp_proto:
            min_required_clusters = 7 * int(self.options.dp_max_proto_per_class)
            if int(self.options.feature_clusters) < min_required_clusters:
                raise ValueError(
                    f"feature_clusters ({self.options.feature_clusters}) is too small for semantic DP prototypes; set --feature_clusters >= {min_required_clusters}."
                )
            self.semantic_dp_manager = SemanticDPPrototypeManager(self.options, self.device)

    def _ensure_depth_options(self):
        defaults = {
            'use_depth_rank_loss': False,
            'use_depth_quantile_loss': False,
            'use_depth_dist_loss': False,
            'depth_rank_weight': 1.0,
            'depth_quantile_weight': 1.0,
            'depth_dist_weight': 1.0,
            'depth_rank_pairs_per_image': 256,
            'min_samples_per_image': 32,
            'depth_quantiles': '0.1,0.5,0.9',
            'depth_normalize': 'zscore',
            'use_depth_bucket_sampling': False,
            'depth_bucket_bins': 4,
            'depth_bucket_ratio': '',
            'depth_bucket_ratio_start': '',
            'depth_bucket_ratio_end': '',
            'depth_bucket_ratio_anneal_epochs': 0,
            'depth_bucket_with_replacement': True,
            'depth_bucket_shuffle_each_epoch': False,
            'depth_bucket_separate_invalid': False,
            'depth_bucket_invalid_ratio': 0.1,
            'use_depth_bucket_quality_weighting': False,
            'depth_bucket_quality_momentum': 0.9,
            'log_depth_bucket_stats': False,
            'use_semantic_dp_proto': False,
            'dp_max_proto_per_class': 16,
            'dp_tau_mode': 'fixed',
            'dp_tau': 2.0,
            'dp_tau_scale': 0.02,
            'dp_proto_ema_beta': 0.9,
            'dp_default_depth': 10.0,
            'dp_use_depth_prior': True,
            'dp_update_every_epochs': 1,
            'use_depth_weighted_repro': False,
            'depth_weighted_epochs': 4,
            'depth_weight_source': 'pred',
            'depth_weight_type': 'inv',
            'depth_weight_scale': 1.0,
        }
        for key, value in defaults.items():
            if not hasattr(self.options, key):
                setattr(self.options, key, value)


    def train(self):
        """
        Main training method.

        Fills a feature buffer using the pretrained encoder and subsequently trains a scene coordinate regression head.
        """

        creating_buffer_time = 0.
        training_time = 0.

        self.training_start = time.time()

        # Create training buffer.
        buffer_start_time = time.time()
        self.create_training_buffer()
        buffer_end_time = time.time()
        creating_buffer_time += buffer_end_time - buffer_start_time
        _logger.info(f"Filled training buffer in {buffer_end_time - buffer_start_time:.1f}s.")

        self.training_buffer['sample_quality'] = torch.ones(
            self.options.training_buffer_size,
            1,
            dtype=torch.float32,
            device=self.device
        )

        # if self.options.feature_clusters > 0:
        if self.options.feature_clusters > 0 and not self.use_semantic_labels:
            self.mini_batch_kmeans.fit(self.training_buffer['features'])
            self.training_buffer['feature_labels'] = self.mini_batch_kmeans.labels_.unsqueeze(-1)

        self._initialize_stage_zero()
        if self.options.use_depth_bucket_sampling:
            self._build_depth_buckets()

        # Train the regression head.
        for self.epoch in range(self.options.epochs):
            epoch_start_time = time.time()
            if self.options.stage_len > 0 and self.epoch > 0 and self.epoch % self.options.stage_len == 0:
                self._update_spatial_clusters()
            if self.options.use_semantic_dp_proto and self.semantic_dp_manager is not None and self.epoch > 0 and self.epoch % int(self.options.dp_update_every_epochs) == 0:
                _logger.info("Semantic DP prototypes status: %s", self.semantic_dp_manager.summary())
            self.run_epoch()
            self._update_buffer_from_predictions()
            training_time += time.time() - epoch_start_time

        # Save trained model.
        self.save_model()

        end_time = time.time()
        _logger.info(f'Done without errors. '
                     f'Creating buffer time: {creating_buffer_time:.1f} seconds. '
                     f'Training time: {training_time:.1f} seconds. '
                     f'Total time: {end_time - self.training_start:.1f} seconds.')

    def create_training_buffer(self):
        # Disable benchmarking, since we have variable tensor sizes.
        torch.backends.cudnn.benchmark = False

        # Sampler.
        batch_sampler = sampler.BatchSampler(
            sampler.RandomSampler(self.dataset, generator=self.batch_generator),
            batch_size=1,
            drop_last=False
        )

        # Used to seed workers in a reproducible manner.
        def seed_worker(worker_id):
            # Different seed per epoch. Initial seed is generated by the main process consuming one random number from
            # the dataloader generator.
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Batching is handled at the dataset level (the dataset __getitem__ receives a list of indices, because we
        # need to rescale all images in the batch to the same size).
        loader_kwargs = {
            'dataset': self.dataset,
            'sampler': batch_sampler,
            'batch_size': None,
            'worker_init_fn': seed_worker,
            'generator': self.loader_generator,
            'pin_memory': self.loader_pin_memory,
            'num_workers': self.num_data_loader_workers,
            'persistent_workers': self.num_data_loader_workers > 0,
        }
        if self.num_data_loader_workers > 0:
            loader_kwargs['prefetch_factor'] = self.loader_prefetch_factor

        training_dataloader = DataLoader(**loader_kwargs)

        _logger.info("Starting creation of the training buffer.")

        # Create a training buffer that lives on the GPU.
        self.training_buffer = {
            'features': torch.empty((self.options.training_buffer_size, self.regressor.feature_dim),
                                    dtype=(torch.float32, torch.float16)[self.options.use_half], device=self.device),
            'target_px': torch.empty((self.options.training_buffer_size, 2), dtype=torch.float32, device=self.device),
            'target_coord': torch.empty((self.options.training_buffer_size, 3), dtype=torch.float32, device=self.device),
            'gt_poses_inv': torch.empty((self.options.training_buffer_size, 3, 4), dtype=torch.float32,
                                        device=self.device),
            'intrinsics': torch.empty((self.options.training_buffer_size, 3, 3), dtype=torch.float32,
                                      device=self.device),
            'intrinsics_inv': torch.empty((self.options.training_buffer_size, 3, 3), dtype=torch.float32,
                                          device=self.device),
            'feature_labels': torch.empty(self.options.training_buffer_size, 1, dtype=torch.float32, device=self.device),
            'spatial_labels': torch.empty(self.options.training_buffer_size, 1, dtype=torch.float32, device=self.device),
            'image_ids': torch.empty(self.options.training_buffer_size, 1, dtype=torch.int64, device=self.device),
            'depth_values': torch.empty(self.options.training_buffer_size, 1, dtype=torch.float32, device=self.device),
            # 'feature_labels': torch.zeros(self.options.training_buffer_size, 1, dtype=torch.float32,
            #                               device=self.device),
            # 'spatial_labels': torch.zeros(self.options.training_buffer_size, 1, dtype=torch.float32,
            #                               device=self.device),
        }

        # Features are computed in evaluation mode.
        self.regressor.eval()

        # The encoder is pretrained, so we don't compute any gradient.
        with torch.no_grad():
            # Iterate until the training buffer is full.
            buffer_idx = 0
            dataset_passes = 0

            iterator = tqdm(total=self.options.training_buffer_size)

            while buffer_idx < self.options.training_buffer_size:
                dataset_passes += 1
                # image_B1HW.size() == image_mask_B1HW.size() 但是他们的size不确定
                # for image_B1HW, image_mask_B1HW, coord_BHW3, gt_pose_B44, gt_pose_inv_B44, intrinsics_B33, intrinsics_inv_B33, _ in training_dataloader:
                for image_B1HW, image_mask_B1HW, coord_BHW3, gt_pose_B44, gt_pose_inv_B44, intrinsics_B33, intrinsics_inv_B33, semantic_BHW, _, depth_B1HW, idx_B in training_dataloader:
                    # Copy to device.
                    image_B1HW = image_B1HW.to(self.device, non_blocking=True)
                    image_mask_B1HW = image_mask_B1HW.to(self.device, non_blocking=True)
                    coord_BHW3 = coord_BHW3.to(self.device, non_blocking=True)
                    gt_pose_inv_B44 = gt_pose_inv_B44.to(self.device, non_blocking=True)
                    intrinsics_B33 = intrinsics_B33.to(self.device, non_blocking=True)
                    intrinsics_inv_B33 = intrinsics_inv_B33.to(self.device, non_blocking=True)
                    if self.use_semantic_labels:
                        semantic_BHW = semantic_BHW.to(self.device, non_blocking = True)
                    depth_B1HW = depth_B1HW.to(self.device, non_blocking=True)
                    idx_B = idx_B.to(self.device, non_blocking=True)

                    # Compute image features.
                    with autocast(enabled=self.options.use_half):
                        # C = 512, features_BCHW.size() = torch.Size([1, 512, 87, 116])
                        # features_BCHW.shape = torch.Size([1, 512, 60, 80])
                        features_BCHW = self.regressor.get_features(image_B1HW)

                    # Dimensions after the network's downsampling.
                    B, C, H, W = features_BCHW.shape

                    # The image_mask needs to be downsampled to the actual output resolution and cast to bool.
                    image_mask_B1HW = TF.resize(image_mask_B1HW, [H, W], interpolation=TF.InterpolationMode.NEAREST)
                    image_mask_B1HW = image_mask_B1HW.bool()

                    # If the current mask has no valid pixels, continue.
                    if image_mask_B1HW.sum() == 0:
                        continue

                    # Create a tensor with the pixel coordinates of every feature vector.
                    pixel_positions_B2HW = self.pixel_grid_2HW[:, :H,
                                           :W].clone()  # It's 2xHxW (actual H and W) now.
                    pixel_positions_B2HW = pixel_positions_B2HW[None]  # 1x2xHxW
                    pixel_positions_B2HW = pixel_positions_B2HW.expand(B, 2, H, W)  # Bx2xHxW

                    # Create a tensor with the coordinates of every feature vector.
                    coord_B3HW = coord_BHW3.permute(0, 3, 1, 2)
                    coord_B3HW = TF.resize(coord_B3HW, [H, W], interpolation=TF.InterpolationMode.NEAREST)
                    if self.use_semantic_labels:
                        semantic_B1HW = semantic_BHW.unsqueeze(1).float()
                        semantic_B1HW = TF.resize(semantic_B1HW, [H, W], interpolation=TF.InterpolationMode.NEAREST)
                    depth_B1HW = TF.resize(depth_B1HW, [H, W], interpolation=TF.InterpolationMode.NEAREST)

                    # Bx3x4 -> Nx3x4 (for each image, repeat pose per feature)
                    gt_pose_inv = gt_pose_inv_B44[:, :3]
                    gt_pose_inv = gt_pose_inv.unsqueeze(1).expand(B, H * W, 3, 4).reshape(-1, 3, 4)

                    # Bx3x3 -> Nx3x3 (for each image, repeat intrinsics per feature)
                    intrinsics = intrinsics_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)
                    intrinsics_inv = intrinsics_inv_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)

                    def normalize_shape(tensor_in):
                        """Bring tensor from shape BxCxHxW to NxC"""
                        return tensor_in.transpose(0, 1).flatten(1).transpose(0, 1)

                    batch_data = {
                        'features': normalize_shape(features_BCHW),
                        'target_px': normalize_shape(pixel_positions_B2HW),
                        'target_coord': normalize_shape(coord_B3HW),
                        'gt_poses_inv': gt_pose_inv,
                        'intrinsics': intrinsics,
                        'intrinsics_inv': intrinsics_inv,
                        'depth_values': normalize_shape(depth_B1HW),
                    }
                    if self.use_semantic_labels:
                        batch_data['feature_labels'] = normalize_shape(semantic_B1HW)
                    image_id_B1HW = idx_B.view(B, 1, 1, 1).expand(B, 1, H, W)
                    batch_data['image_ids'] = normalize_shape(image_id_B1HW).to(torch.int64)

                    # Turn image mask into sampling weights (all equal).
                    image_mask_B1HW = image_mask_B1HW.float()
                    image_mask_N1 = normalize_shape(image_mask_B1HW)

                    # Over-sample according to image mask.
                    features_to_select = self.options.samples_per_image * B
                    features_to_select = min(features_to_select, self.options.training_buffer_size - buffer_idx)

                    # Sample indices uniformly, with replacement.
                    sample_idxs = torch.multinomial(
                        image_mask_N1.view(-1),
                        features_to_select,
                        replacement=True,
                        generator=self.sampling_generator
                    )

                    # Select the data to put in the buffer.
                    for k in batch_data:
                        batch_data[k] = batch_data[k][sample_idxs]

                    if self.options.use_semantic_dp_proto and self.use_semantic_labels and self.semantic_dp_manager is not None:
                        batch_data['feature_labels'] = self.semantic_dp_manager.assign(
                            batch_data['feature_labels'],
                            batch_data['target_px'],
                            batch_data['intrinsics_inv'],
                            batch_data['gt_poses_inv'],
                            batch_data['depth_values']
                        )

                    # def finite_mask_for_batch(batch_data):
                    #     m = None
                    #     for k, v in batch_data.items():
                    #         if not torch.is_tensor(v):
                    #             continue
                    #         cur = torch.isfinite(v).all(dim=1) if v.ndim == 2 else torch.isfinite(v).view(v.shape[0],
                    #                                                                                       -1).all(dim=1)
                    #         m = cur if m is None else (m & cur)
                    #     return m
                    #
                    # finite_m = finite_mask_for_batch(batch_data)
                    # if finite_m.sum() == 0:
                    #     continue
                    # for k in batch_data:
                    #     batch_data[k] = batch_data[k][finite_m]
                    # features_to_select = batch_data['features'].shape[0]

                    # Write to training buffer. Start at buffer_idx and end at buffer_offset - 1.
                    buffer_offset = buffer_idx + features_to_select
                    for k in batch_data:
                        self.training_buffer[k][buffer_idx:buffer_offset] = batch_data[k]

                    buffer_idx = buffer_offset
                    iterator.update(features_to_select)
                    if buffer_idx >= self.options.training_buffer_size:
                        break

        buffer_memory = sum([v.element_size() * v.nelement() for k, v in self.training_buffer.items()])
        buffer_memory /= 1024 * 1024 * 1024

        _logger.info(f"Created buffer of {buffer_memory:.2f}GB with {dataset_passes} passes over the training data.")
        self.regressor.train()

    def _initialize_stage_zero(self):
        _logger.info("Initializing stage 0: zero spatial labels and centers.")
        self.training_buffer['spatial_labels'].zero_()
        self.regressor.heads.cluster_centers.zero_()

    def _update_spatial_clusters(self):
        if self.options.spatial_clusters <= 0:
            return
        _logger.info("Updating spatial clusters from buffer coordinates.")
        points = self.training_buffer['target_coord']
        centers, _, labels = self.generate_clusters(self.options.spatial_clusters, points)
        self.training_buffer['spatial_labels'] = torch.tensor(labels).to(self.device).unsqueeze(-1).to(torch.float32)
        new_centers = self._compute_cluster_centers()
        if self.options.use_ema_centers:
            beta = float(self.options.ema_beta)
            new_centers = beta * self.regressor.heads.cluster_centers + (1.0 - beta) * new_centers
        self.regressor.heads.cluster_centers = new_centers

    def _compute_cluster_centers(self):
        centers = []
        for i in range(self.options.feature_clusters):
            for j in range(self.options.spatial_clusters):
                mask1 = self.training_buffer['feature_labels'] == i
                mask2 = self.training_buffer['spatial_labels'] == j
                mask = mask1 * mask2
                if not torch.any(mask):
                    centers.append(self.regressor.heads.mean.flatten().to(torch.float32).unsqueeze(-1))
                    continue
                center = torch.mean(self.training_buffer['target_coord'][mask.squeeze(-1)], dim=0)
                _logger.info(f"     Center {i:3d}/{j:3d}: {center}")
                centers.append(center.unsqueeze(-1).to(torch.float32))
        centers = torch.concatenate(centers, dim=1)
        return centers[None][None].permute(3, 2, 0, 1).to(self.device).clone().detach()

    def _update_buffer_from_predictions(self):
        _logger.info("Updating buffer target coordinates from model predictions.")
        self.regressor.eval()
        batch_size = self.options.batch_size
        end_index = self.options.training_buffer_size
        end_index -= end_index % batch_size
        with torch.no_grad():
            for batch_start in range(0, end_index, batch_size):
                batch_end = min(batch_start + batch_size, end_index)
                features_bC = self.training_buffer['features'][batch_start:batch_end].contiguous()
                feature_labels = self.training_buffer['feature_labels'][batch_start:batch_end].contiguous()
                spatial_labels = self.training_buffer['spatial_labels'][batch_start:batch_end].contiguous()
                pred_coords = self._predict_scene_coords(features_bC, feature_labels, spatial_labels)
                self.training_buffer['target_coord'][batch_start:batch_end] = pred_coords
        self.regressor.train()

    def _predict_scene_coords(self, features_bC, feature_labels, spatial_labels):
        batch_size = features_bC.shape[0]
        channels = features_bC.shape[1]
        features_bCHW = features_bC[None, None, ...].view(-1, 16, 32, channels).permute(0, 3, 1, 2)
        Kf = int(self.options.feature_clusters)
        Ks = int(self.options.spatial_clusters)
        feature_labels_onehot = None
        if Kf > 1:
            feature_labels_onehot = torch.nn.functional.one_hot(
                feature_labels.view(-1).to(torch.int64).clamp(0, Kf - 1),
                num_classes=Kf
            ).to(self.device, torch.float32)
            feature_labels_onehot = feature_labels_onehot[None, None, ...].view(-1, 16, 32, Kf).permute(0, 3, 1,
                                                                                                        2).contiguous()
        spatial_labels_onehot = None
        if Ks > 1:
            spatial_labels_onehot = torch.nn.functional.one_hot(
                spatial_labels.view(-1).to(torch.int64).clamp(0, Ks - 1),
                num_classes=Ks
            ).to(self.device, torch.float32)
            spatial_labels_onehot = spatial_labels_onehot[None, None, ...].view(-1, 16, 32, Ks).permute(0, 3, 1,
                                                                                                        2).contiguous()
        with autocast(enabled=self.options.use_half):
            pred_scene_coords, _, _, _ = self.regressor.get_scene_coordinates(
                features_bCHW, feature_labels_onehot, spatial_labels_onehot
            )
        pred_scene_coords_b3HW = pred_scene_coords[:, 0:3, :, :]
        pred_scene_coords_b31 = pred_scene_coords_b3HW.permute(0, 2, 3, 1).flatten(0, 2).float()
        return pred_scene_coords_b31[:batch_size]

    def _build_depth_buckets(self):
        _logger.info("Building depth buckets for balanced sampling.")
        depth_values = self.training_buffer['depth_values'].view(-1).detach().cpu()
        valid_mask = depth_values > 0
        valid_depths = depth_values[valid_mask]
        num_bins = int(self.options.depth_bucket_bins)
        if valid_depths.numel() == 0:
            self.depth_bucket_indices = [torch.arange(depth_values.numel())]
            self.depth_bucket_cursors = [0]
            return

        quantiles = torch.linspace(0, 1, num_bins + 1)
        edges = torch.quantile(valid_depths, quantiles)
        if torch.unique(edges).numel() < edges.numel():
            min_val = valid_depths.min()
            max_val = valid_depths.max()
            edges = torch.linspace(min_val, max_val, num_bins + 1)

        bucket_ids = torch.bucketize(depth_values, edges[1:-1], right=True)
        bucket_ids = bucket_ids.clamp(max=num_bins - 1)
        if self.options.depth_bucket_separate_invalid:
            bucket_ids[~valid_mask] = num_bins
            num_bins += 1
        else:
            bucket_ids[~valid_mask] = num_bins - 1

        self.depth_bucket_indices = []
        self.depth_bucket_cursors = []
        for bin_idx in range(num_bins):
            indices = torch.nonzero(bucket_ids == bin_idx, as_tuple=False).view(-1)
            if indices.numel() == 0:
                indices = torch.empty((0,), dtype=torch.long)
            self.depth_bucket_indices.append(indices)
            self.depth_bucket_cursors.append(0)
        self.depth_bucket_last_epoch = None

    def _parse_ratio_string(self, ratio_value, num_bins):
        if ratio_value:
            parts = [float(x) for x in ratio_value.split(',') if x.strip() != '']
        else:
            parts = []
        if not parts:
            return [1.0] * num_bins
        if len(parts) != num_bins:
            _logger.warning("depth_bucket_ratio length does not match depth_bucket_bins; falling back to uniform.")
            return [1.0] * num_bins
        return parts

    def _parse_bucket_ratios(self):
        num_bins = len(self.depth_bucket_indices) if self.depth_bucket_indices else int(self.options.depth_bucket_bins)
        if self.options.depth_bucket_separate_invalid and self.options.depth_bucket_ratio:
            base_bins = max(num_bins - 1, 1)
            ratios = self._parse_ratio_string(self.options.depth_bucket_ratio, base_bins)
            if len(ratios) == base_bins:
                ratios = list(ratios) + [float(self.options.depth_bucket_invalid_ratio)]
            else:
                ratios = self._parse_ratio_string(self.options.depth_bucket_ratio, num_bins)
        else:
            ratios = self._parse_ratio_string(self.options.depth_bucket_ratio, num_bins)

        if self.options.depth_bucket_ratio_anneal_epochs > 0:
            start_ratios = self._parse_ratio_string(self.options.depth_bucket_ratio_start, num_bins)
            end_ratios = self._parse_ratio_string(self.options.depth_bucket_ratio_end, num_bins)
            denom = float(max(self.options.depth_bucket_ratio_anneal_epochs - 1, 1))
            t = min(self.epoch / denom, 1.0)
            ratios = [
                (1 - t) * s + t * e
                for s, e in zip(start_ratios, end_ratios)
            ]

        if self.options.depth_bucket_separate_invalid and self.options.depth_bucket_invalid_ratio >= 0:
            if num_bins > 1:
                ratios = list(ratios)
                ratios[-1] = float(self.options.depth_bucket_invalid_ratio)
        return ratios

    def _depth_bucket_batch_indices(self, batch_size):
        if not self.depth_bucket_indices:
            return torch.randint(0, self.options.training_buffer_size, (batch_size,), generator=self.training_generator)

        ratios = self._parse_bucket_ratios()
        ratio_sum = sum(ratios)
        counts = [int(batch_size * r / ratio_sum) for r in ratios]
        remainder = batch_size - sum(counts)
        for i in range(remainder):
            counts[i % len(counts)] += 1

        batch_indices = []
        for bin_idx, count in enumerate(counts):
            if count <= 0:
                continue
            indices = self.depth_bucket_indices[bin_idx]
            if indices.numel() == 0:
                fallback = torch.randint(0, self.options.training_buffer_size, (count,), generator=self.training_generator)
                batch_indices.append(fallback)
                continue
            cursor = self.depth_bucket_cursors[bin_idx]
            remaining = indices.numel() - cursor
            if self.options.use_depth_bucket_quality_weighting:
                device_indices = indices.to(self.device)
                weights = self.training_buffer['sample_quality'][device_indices].view(-1).float()
                if weights.sum() <= 0:
                    weights = torch.ones_like(weights)
                if count >= indices.numel():
                    chosen = device_indices
                else:
                    chosen = device_indices[torch.multinomial(
                        weights,
                        count,
                        replacement=self.options.depth_bucket_with_replacement
                    )]
                selected = chosen.detach().cpu()
            else:
                if remaining >= count:
                    selected = indices[cursor:cursor + count]
                    cursor += count
                else:
                    first = indices[cursor:]
                    needed = count - remaining
                    cursor = 0
                    if needed > 0:
                        if self.options.depth_bucket_with_replacement:
                            rand_idx = torch.randint(0, indices.numel(), (needed,), generator=self.depth_bucket_generator)
                            second = indices[rand_idx]
                        else:
                            second = indices[:min(needed, indices.numel())]
                        selected = torch.cat([first, second], dim=0)
                        cursor = min(needed, indices.numel())
                    else:
                        selected = first
            self.depth_bucket_cursors[bin_idx] = cursor
            batch_indices.append(selected)

        if not batch_indices:
            return torch.randint(0, self.options.training_buffer_size, (batch_size,), generator=self.training_generator)
        batch_indices = torch.cat(batch_indices, dim=0)
        if batch_indices.numel() > batch_size:
            perm = torch.randperm(batch_indices.numel(), generator=self.training_generator)
            batch_indices = batch_indices[perm[:batch_size]]
        elif batch_indices.numel() < batch_size:
            extra = torch.randint(0, self.options.training_buffer_size, (batch_size - batch_indices.numel(),),
                                  generator=self.training_generator)
            batch_indices = torch.cat([batch_indices, extra], dim=0)
        return batch_indices

    def _maybe_shuffle_depth_buckets(self):
        if not self.options.depth_bucket_shuffle_each_epoch:
            return
        if self.depth_bucket_indices is None:
            return
        if self.depth_bucket_last_epoch == self.epoch:
            return
        for bin_idx, indices in enumerate(self.depth_bucket_indices):
            if indices.numel() > 1:
                perm = torch.randperm(indices.numel(), generator=self.depth_bucket_generator)
                self.depth_bucket_indices[bin_idx] = indices[perm]
            self.depth_bucket_cursors[bin_idx] = 0
        self.depth_bucket_last_epoch = self.epoch

    def _log_depth_bucket_stats(self):
        if not self.depth_bucket_indices:
            return
        depth_values = self.training_buffer['depth_values'].view(-1).detach().cpu()
        quality_values = self.training_buffer['sample_quality'].view(-1).detach().cpu()
        stats = []
        for bin_idx, indices in enumerate(self.depth_bucket_indices):
            if indices.numel() == 0:
                stats.append(f"bucket{bin_idx}:0")
                continue
            bucket_depths = depth_values[indices]
            bucket_quality = quality_values[indices]
            valid_ratio = float((bucket_depths > 0).float().mean().item())
            quality_mean = float(bucket_quality.mean().item())
            stats.append(f"bucket{bin_idx}:{indices.numel()}(valid={valid_ratio:.2f},q={quality_mean:.3f})")
        _logger.info("Depth bucket stats: %s", " ".join(stats))

    def _compute_depth_weight(self, pred_cam_coords_b31, depth_values_b1):
        if not self.options.use_depth_weighted_repro:
            return None
        if self.options.depth_weighted_epochs <= 0:
            return None
        if self.epoch >= self.options.depth_weighted_epochs:
            return None

        if self.options.depth_weight_source == 'gt':
            if not self.use_depth:
                return None
            depth_source = depth_values_b1.view(-1)
            valid_mask = depth_source > 0
            depth_source = torch.where(
                valid_mask,
                depth_source,
                torch.full_like(depth_source, self.options.depth_min)
            )
        else:
            depth_source = pred_cam_coords_b31[:, 2].detach().squeeze(-1)
            valid_mask = torch.ones_like(depth_source, dtype=torch.bool)

        depth_source = depth_source.clamp(min=self.options.depth_min)
        if self.options.depth_weight_type == 'log':
            denom = torch.log1p(depth_source).clamp(min=1e-6)
            weight = self.options.depth_weight_scale / denom
        else:
            weight = self.options.depth_weight_scale / depth_source

        if self.options.depth_weight_source == 'gt':
            weight = torch.where(valid_mask, weight, torch.ones_like(weight))

        return weight.unsqueeze(-1)

    def _compute_depth_losses(self, pred_depth_b1, depth_values_b1, image_ids_b1):
        if not self.use_depth:
            return torch.zeros((), device=self.device), torch.zeros((), device=self.device), torch.zeros((), device=self.device)
        if not (self.options.use_depth_rank_loss or self.options.use_depth_quantile_loss or self.options.use_depth_dist_loss):
            return torch.zeros((), device=self.device), torch.zeros((), device=self.device), torch.zeros((), device=self.device)

        image_ids = image_ids_b1.view(-1).to(torch.int64)
        pred_depth = pred_depth_b1.view(-1)
        depth_values = depth_values_b1.view(-1)
        valid_mask = depth_values > 0
        image_ids = image_ids[valid_mask]
        pred_depth = pred_depth[valid_mask]
        depth_values = depth_values[valid_mask]

        unique_ids, counts = torch.unique(image_ids, return_counts=True)
        eligible_ids = unique_ids[counts >= self.options.min_samples_per_image]

        if eligible_ids.numel() == 0:
            return torch.zeros((), device=self.device), torch.zeros((), device=self.device), torch.zeros((), device=self.device)

        rank_losses = []
        quantile_losses = []
        dist_losses = []
        q_tensor = torch.tensor(self.depth_quantiles, device=self.device, dtype=torch.float32)

        for img_id in eligible_ids:
            mask = image_ids == img_id
            z_pred = pred_depth[mask]
            z_rel = depth_values[mask]

            if self.options.depth_normalize == 'zscore':
                z_pred = (z_pred - z_pred.mean()) / (z_pred.std() + 1e-6)
                z_rel = (z_rel - z_rel.mean()) / (z_rel.std() + 1e-6)
            elif self.options.depth_normalize == 'minmax':
                z_pred = (z_pred - z_pred.min()) / (z_pred.max() - z_pred.min() + 1e-6)
                z_rel = (z_rel - z_rel.min()) / (z_rel.max() - z_rel.min() + 1e-6)

            if self.options.use_depth_rank_loss:
                m = z_pred.shape[0]
                num_pairs = min(self.options.depth_rank_pairs_per_image, m // 2)
                if num_pairs > 0:
                    perm = torch.randperm(m, device=self.device)
                    idx_i = perm[:num_pairs]
                    idx_j = perm[num_pairs:2 * num_pairs]
                    sign = torch.sign(z_rel[idx_i] - z_rel[idx_j])
                    valid = sign != 0
                    if valid.any():
                        diff = z_pred[idx_i] - z_pred[idx_j]
                        rank_loss = torch.nn.functional.softplus(-sign[valid] * diff[valid]).mean()
                        rank_losses.append(rank_loss)

            if self.options.use_depth_quantile_loss:
                if z_pred.numel() >= 2:
                    q_pred = torch.quantile(z_pred, q_tensor)
                    q_rel = torch.quantile(z_rel, q_tensor)
                    quantile_losses.append(torch.nn.functional.l1_loss(q_pred, q_rel))

            if self.options.use_depth_dist_loss:
                if z_pred.numel() >= 2:
                    z_pred_sorted, _ = torch.sort(z_pred)
                    z_rel_sorted, _ = torch.sort(z_rel)
                    dist_losses.append(torch.nn.functional.l1_loss(z_pred_sorted, z_rel_sorted))

        rank_loss = torch.stack(rank_losses).mean() if rank_losses else torch.zeros((), device=self.device)
        quantile_loss = torch.stack(quantile_losses).mean() if quantile_losses else torch.zeros((), device=self.device)
        dist_loss = torch.stack(dist_losses).mean() if dist_losses else torch.zeros((), device=self.device)
        return rank_loss, quantile_loss, dist_loss

    def generate_clusters(self, num_clusters, points):
        num_points = points.shape[0]
        _logger.info(f'Clustering a dataset with {num_points} frames into {num_clusters} clusters.')

        # A tensor holding all camera centers used for clustering.
        cam_centers = points.cpu().detach().numpy()

        # Setup kMEans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        flags = cv2.KMEANS_PP_CENTERS

        # Label of next cluster.
        label_counter = 0

        # Initialise list of clusters with all images.
        clusters = []
        clusters.append((cam_centers, label_counter, np.zeros(3)))

        # All images belong to cluster 0.
        labels = np.zeros(num_points)

        # iterate kMeans with k=2
        while len(clusters) < num_clusters:
            # Select largest cluster (list is sorted).
            cur_cluster = clusters.pop(0)
            label_counter += 1

            # Split cluster.
            cur_error, cur_labels, cur_centroids = cv2.kmeans(cur_cluster[0], 2, None, criteria, 10, flags)

            # Update cluster list.
            cur_mask = (cur_labels == 0)[:, 0]
            cur_cam_centers0 = cur_cluster[0][cur_mask, :]
            clusters.append((cur_cam_centers0, cur_cluster[1], cur_centroids[0]))

            cur_mask = (cur_labels == 1)[:, 0]
            cur_cam_centers1 = cur_cluster[0][cur_mask, :]
            clusters.append((cur_cam_centers1, label_counter, cur_centroids[1]))

            cluster_labels = labels[labels == cur_cluster[1]]
            cluster_labels[cur_mask] = label_counter
            labels[labels == cur_cluster[1]] = cluster_labels

            # Sort updated list.
            clusters = sorted(clusters, key=lambda cluster: cluster[0].shape[0], reverse=True)

        # clusters are sorted but cluster indices are random, remap cluster indices to sorted indices
        remapped_labels = np.zeros(num_points)
        remapped_clusters = []

        for cluster_idx_new, cluster in enumerate(clusters):
            cluster_idx_old = cluster[1]
            remapped_labels[labels == cluster_idx_old] = cluster_idx_new
            remapped_clusters.append((cluster[0], cluster_idx_new, cluster[2]))

        labels = remapped_labels
        clusters = remapped_clusters

        cluster_centers = np.zeros((num_clusters, 3))
        cluster_sizes = np.zeros((num_clusters, 1))

        for cluster in clusters:
            # Compute distance of each cam to the center of the cluster.
            cam_num = cluster[0].shape[0]
            cam_data = np.zeros((cam_num, 3))
            cam_count = 0

            # First compute the center of the cluster (mean).
            for i, cam_center in enumerate(cam_centers):
                if labels[i] == cluster[1]:
                    cam_data[cam_count] = cam_center
                    cam_count += 1

            cluster_centers[cluster[1]] = cam_data.mean(0)

            # Compute the distance of each cam from the cluster center. Then average and square.
            cam_dists = np.broadcast_to(cluster_centers[cluster[1]][np.newaxis, :], (cam_num, 3))
            cam_dists = cam_data - cam_dists
            cam_dists = np.linalg.norm(cam_dists, axis=1)
            cam_dists = cam_dists ** 2

            cluster_sizes[cluster[1]] = cam_dists.mean()

            _logger.info("Cluster %i: %.1fm, %.1fm, %.1fm, images: %i, mean squared dist: %f" % (
                cluster[1], cluster_centers[cluster[1]][0], cluster_centers[cluster[1]][1],
                cluster_centers[cluster[1]][2],
                cluster[0].shape[0], cluster_sizes[cluster[1]]))

        _logger.info('Clustering done.')

        return cluster_centers, cluster_sizes, labels

    def run_epoch(self):
        """
        Run one epoch of training, shuffling the feature buffer and iterating over it.
        """
        # Enable benchmarking since all operations work on the same tensor size.
        torch.backends.cudnn.benchmark = True

        # Shuffle indices.
        end_index = self.options.training_buffer_size
        if not self.options.use_depth_bucket_sampling:
            random_indices = torch.randperm(end_index, generator=self.training_generator)
        else:
            random_indices = None
            self._maybe_shuffle_depth_buckets()
            if self.options.log_depth_bucket_stats:
                self._log_depth_bucket_stats()

        # Iterate with mini batches.
        for batch_start in range(0, end_index, self.options.batch_size):
            batch_end = batch_start + self.options.batch_size

            # Drop last batch if not full.
            if batch_end > end_index:
                continue

            # Sample indices.
            if self.options.use_depth_bucket_sampling:
                random_batch_indices = self._depth_bucket_batch_indices(self.options.batch_size)
            else:
                random_batch_indices = random_indices[batch_start:batch_end]

            # Call the training step with the sampled features and relevant metadata.
            self.training_step(
                self.training_buffer['features'][random_batch_indices].contiguous(),
                self.training_buffer['target_px'][random_batch_indices].contiguous(),
                self.training_buffer['target_coord'][random_batch_indices].contiguous(),
                self.training_buffer['gt_poses_inv'][random_batch_indices].contiguous(),
                self.training_buffer['intrinsics'][random_batch_indices].contiguous(),
                self.training_buffer['intrinsics_inv'][random_batch_indices].contiguous(),
                self.training_buffer['feature_labels'][random_batch_indices].contiguous(),
                self.training_buffer['spatial_labels'][random_batch_indices].contiguous(),
                self.training_buffer['image_ids'][random_batch_indices].contiguous(),
                self.training_buffer['depth_values'][random_batch_indices].contiguous(),
                random_batch_indices,
            )
            self.iteration += 1

    def training_step(self, features_bC, target_px_b2, target_coord, gt_inv_poses_b34, Ks_b33, invKs_b33, feature_labels, spatial_labels, image_ids, depth_values, sample_indices):
        """
        Run one iteration of training, computing the reprojection error and minimising it.
        """

        # def assert_finite(name, x):
        #     if not torch.isfinite(x).all():
        #         raise RuntimeError(f"{name} contains NaN/Inf")
        #
        # assert_finite("features_bC", features_bC)
        # assert_finite("target_px_b2", target_px_b2)
        # assert_finite("target_coord", target_coord)
        # assert_finite("Ks_b33", Ks_b33)
        # assert_finite("gt_inv_poses_b34", gt_inv_poses_b34)


        batch_size = features_bC.shape[0]
        channels = features_bC.shape[1]

        # Reshape to a "fake" BCHW shape, since it's faster to run through the network compared to the original shape.
        features_bCHW = features_bC[None, None, ...].view(-1, 16, 32, channels).permute(0, 3, 1, 2)

        # cluster_number = self.options.feature_clusters
        # feature_labels_onehot = torch.FloatTensor(cluster_number, feature_labels.size()[0]).zero_().to('cuda')
        # feature_labels_onehot = feature_labels_onehot.scatter_(0, feature_labels.flatten().unsqueeze(0).to(torch.int64), 1).permute(1, 0)
        # feature_labels_onehot = feature_labels_onehot[None, None, ...].view(-1, 16, 32, cluster_number).permute(0, 3, 1, 2)
        #
        # cluster_number = self.options.spatial_clusters
        # spatial_labels_onehot = torch.FloatTensor(cluster_number, spatial_labels.size()[0]).zero_().to('cuda')
        # spatial_labels_onehot = spatial_labels_onehot.scatter_(0, spatial_labels.flatten().unsqueeze(0).to(torch.int64), 1).permute(1, 0)
        # spatial_labels_onehot = spatial_labels_onehot[None, None, ...].view(-1, 16, 32, cluster_number).permute(0, 3, 1, 2)

        Kf = int(self.options.feature_clusters)
        Ks = int(self.options.spatial_clusters)

        feature_labels_onehot = None
        if Kf > 1:
            feature_labels_onehot = torch.nn.functional.one_hot(
                feature_labels.view(-1).to(torch.int64).clamp(0, Kf - 1),
                num_classes=Kf
            ).to(self.device, torch.float32)
            feature_labels_onehot = feature_labels_onehot[None, None, ...].view(-1, 16, 32, Kf).permute(0, 3, 1,
                                                                                                        2).contiguous()

        spatial_labels_onehot = None
        if Ks > 1:
            spatial_labels_onehot = torch.nn.functional.one_hot(
                spatial_labels.view(-1).to(torch.int64).clamp(0, Ks - 1),
                num_classes=Ks
            ).to(self.device, torch.float32)
            spatial_labels_onehot = spatial_labels_onehot[None, None, ...].view(-1, 16, 32, Ks).permute(0, 3, 1,
                                                                                                        2).contiguous()

        with autocast(enabled=self.options.use_half):
            pred_scene_coords, pred_spatial_labels, pred_feature_labels, _ = self.regressor.get_scene_coordinates(features_bCHW, feature_labels_onehot, spatial_labels_onehot)

        # Back to the original shape. Convert to float32 as well.
        pred_scene_coords_b3HW = pred_scene_coords[:, 0:3, :, :]
        pred_scene_coords_b31 = pred_scene_coords_b3HW.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()

        # pred_spatial_labels = pred_spatial_labels.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()
        # pred_spatial_labels = pred_spatial_labels.permute(2, 1, 0)
        # spatial_labels = spatial_labels.permute(1, 0).to(torch.int64)
        # spatial_labels_loss = self.cls_loss(pred_spatial_labels, spatial_labels)
        #
        # pred_feature_labels = pred_feature_labels.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()
        # pred_feature_labels = pred_feature_labels.permute(2, 1, 0)
        # feature_labels = feature_labels.permute(1, 0).to(torch.int64)
        # feature_labels_loss = self.cls_loss(pred_feature_labels, feature_labels)

        # spatial loss
        spatial_loss_active = Ks > 1 and (
            self.options.stage_len <= 0 or self.epoch >= self.options.stage_len
        )
        if spatial_loss_active:
            spatial_logits = pred_spatial_labels.permute(0, 2, 3, 1).reshape(-1, Ks).float()
            spatial_tgt = spatial_labels.view(-1).to(torch.int64).clamp(0, Ks - 1)
            spatial_labels_loss = F.cross_entropy(spatial_logits, spatial_tgt)
        else:
            spatial_labels_loss = torch.zeros((), device=self.device, dtype=torch.float32)

        # feature loss
        if Kf > 1:
            feature_logits = pred_feature_labels.permute(0, 2, 3, 1).reshape(-1, Kf).float()
            feature_tgt = feature_labels.view(-1).to(torch.int64).clamp(0, Kf - 1)
            feature_labels_loss = F.cross_entropy(feature_logits, feature_tgt)
        else:
            feature_labels_loss = torch.zeros((), device=self.device, dtype=torch.float32)

        pred_scene_conf_b1HW = pred_scene_coords[:, -1, :, :]
        pred_scene_conf_b1 = pred_scene_conf_b1HW.flatten(0, 2).unsqueeze(-1).float()
        if self.options.use_depth_bucket_quality_weighting and sample_indices is not None:
            momentum = float(self.options.depth_bucket_quality_momentum)
            current = self.training_buffer['sample_quality'][sample_indices].view(-1, 1)
            updated = current * momentum + pred_scene_conf_b1.detach() * (1.0 - momentum)
            self.training_buffer['sample_quality'][sample_indices] = updated

        # Make 3D points homogeneous so that we can easily matrix-multiply them.
        pred_scene_coords_b41 = to_homogeneous(pred_scene_coords_b31)

        # Scene coordinates to camera coordinates.
        pred_cam_coords_b31 = torch.bmm(gt_inv_poses_b34, pred_scene_coords_b41)

        # Project scene coordinates.
        pred_px_b31 = torch.bmm(Ks_b33, pred_cam_coords_b31)

        # Avoid division by zero.
        # Note: negative values are also clamped at +self.options.depth_min. The predicted pixel would be wrong,
        # but that's fine since we mask them out later.
        pred_px_b31[:, 2].clamp_(min=self.options.depth_min)

        # Dehomogenise.
        pred_px_b21 = pred_px_b31[:, :2] / pred_px_b31[:, 2, None]

        # Measure reprojection error.
        reprojection_error_b2 = pred_px_b21.squeeze() - target_px_b2
        reprojection_error_b1 = torch.norm(reprojection_error_b2, dim=1, keepdim=True, p=1)

        depth_weight_b1 = self._compute_depth_weight(pred_cam_coords_b31, depth_values)
        if depth_weight_b1 is not None:
            reprojection_error_b1 = reprojection_error_b1 * depth_weight_b1

        #
        # Compute masks used to ignore invalid pixels.
        #
        # Predicted coordinates behind or close to camera plane.

        reprojection_error_b1 = (pred_scene_conf_b1 * reprojection_error_b1) - self.conf_alpha * torch.log(pred_scene_conf_b1)

        invalid_min_depth_b1 = pred_cam_coords_b31[:, 2] < self.options.depth_min
        # Very large reprojection errors.
        invalid_repro_b1 = reprojection_error_b1 > self.options.repro_loss_hard_clamp
        # Predicted coordinates beyond max distance.
        invalid_max_depth_b1 = pred_cam_coords_b31[:, 2] > self.options.depth_max

        # Invalid mask is the union of all these. Valid mask is the opposite.
        invalid_mask_b1 = (invalid_min_depth_b1 | invalid_repro_b1 | invalid_max_depth_b1)
        valid_mask_b1 = ~invalid_mask_b1

        # # Reprojection error for all valid scene coordinates.
        valid_reprojection_error_b1 = reprojection_error_b1[valid_mask_b1]
        # # Compute the loss for valid predictions.
        loss_valid = self.repro_loss.compute(valid_reprojection_error_b1, self.iteration)

        # Handle the invalid predictions: generate proxy coordinate targets with constant depth assumption.
        pixel_grid_crop_b31 = to_homogeneous(target_px_b2.unsqueeze(2))
        target_camera_coords_b31 = self.options.depth_target * torch.bmm(invKs_b33, pixel_grid_crop_b31)

        # Compute the distance to target camera coordinates.
        invalid_mask_b11 = invalid_mask_b1.unsqueeze(2)
        loss_invalid = torch.abs(target_camera_coords_b31 - pred_cam_coords_b31).masked_select(
            invalid_mask_b11).sum()

        # Final loss is the sum of all 2.
        loss_repro = loss_valid + loss_invalid
        loss_repro /= batch_size

        pred_depth_b1 = pred_cam_coords_b31[:, 2].squeeze(-1)
        rank_loss, quantile_loss, dist_loss = self._compute_depth_losses(
            pred_depth_b1,
            depth_values,
            image_ids
        )
        depth_loss = (
            rank_loss * self.options.depth_rank_weight
            + quantile_loss * self.options.depth_quantile_weight
            + dist_loss * self.options.depth_dist_weight
        )

        # conf judge (only when Ks > 1)
        if Ks > 1 and self.use_depth and self.options.use_depth_for_coord:
            loss_euc = self.euclidean_loss.compute(
                target_coord,
                pred_scene_coords_b31.squeeze(),
                None,
                valid_mask_b1
            )
        else:
            loss_euc = torch.zeros((), device=self.device, dtype=torch.float32)

        loss = loss_euc * self.alpha + loss_repro + spatial_labels_loss * 100 + feature_labels_loss * 100 + depth_loss

        # We need to check if the step actually happened, since the scaler might skip optimisation steps.
        old_optimizer_step = self.optimizer._step_count

        # Optimization steps.
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward(retain_graph=True)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.iteration % self.iterations_output == 0:
            # Print status.
            time_since_start = time.time() - self.training_start
            fraction_valid = float(valid_mask_b1.sum() / batch_size)

            _logger.info(f'Iteration: {self.iteration:6d} / Epoch {self.epoch:03d}|{self.options.epochs:03d}, '
                         f'Loss: {loss:.1f}, '
                         f'Euc Loss: {loss_euc:.1f}, '
                         f'Rep Loss: {loss_repro:.1f}, '
                         f'Valid Loss: {loss_valid:.1f}, '
                         f'Invalid Loss: {loss_invalid:.1f}, '
                         f'Spatial Loss: {spatial_labels_loss:.2f}, '
                         f'Feature Loss: {feature_labels_loss:.2f}, '
                         f'Depth Loss: {depth_loss:.2f}, '
                         f'Rank Loss: {rank_loss:.2f}, '
                         f'Quantile Loss: {quantile_loss:.2f}, '
                         f'Dist Loss: {dist_loss:.2f}, '
                         f'Valid: {fraction_valid * 100:.1f}%, '
                         f'Time: {time_since_start:.2f}s')

        # Only step if the optimizer stepped and if we're not over-stepping the total_steps supported by the scheduler.
        if old_optimizer_step < self.optimizer._step_count < self.scheduler.total_steps:
            self.scheduler.step()

    def save_model(self):
        # NOTE: This would save the whole regressor (encoder weights included) in full precision floats (~30MB).
        # torch.save(self.regressor.state_dict(), self.options.output_map_file)

        # This saves just the head weights as half-precision floating point numbers for a total of ~4MB, as mentioned
        # in the paper. The scene-agnostic encoder weights can then be loaded from the pretrained encoder file.
        head_state_dict = self.regressor.heads.state_dict()
        for k, v in head_state_dict.items():
            head_state_dict[k] = head_state_dict[k].half()
        torch.save(head_state_dict, self.options.output_map_file)
        _logger.info(f"Saved trained head weights to: {self.options.output_map_file}")
