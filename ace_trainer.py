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

        self.device = torch.device('cuda')

        # The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
        # torch.backends.cuda.matmul.allow_tf32 = False

        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        # torch.backends.cudnn.allow_tf32 = False

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
        self.num_data_loader_workers = 12

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
        )
        self.use_semantic_labels = self.dataset.semantic_files is not None

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

        # Will be filled at the beginning of the training process.
        self.training_buffer = None
        self.conf_alpha = 20
        self.alpha = 0.1
        self.mini_batch_kmeans = MiniBatchKMeansCUDA(n_clusters=self.options.feature_clusters, batch_size=400000, max_iter=200, device='cuda')


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

        # if self.options.feature_clusters > 0:
        if self.options.feature_clusters > 0 and not self.use_semantic_labels:
            self.mini_batch_kmeans.fit(self.training_buffer['features'])
            self.training_buffer['feature_labels'] = self.mini_batch_kmeans.labels_.unsqueeze(-1)
        if self.options.spatial_clusters > 0:
            points = self.training_buffer['target_coord']
            centers, _, labels = self.generate_clusters(self.options.spatial_clusters, points)
            self.training_buffer['spatial_labels'] = torch.tensor(labels).to('cuda').unsqueeze(-1).to(torch.float32)
            # centers = torch.tensor(centers)[None][None].permute(2, 3, 0, 1).to('cuda')
            # self.regressor.heads.cluster_centers = centers

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
        centers = centers[None][None].permute(3, 2, 0, 1).to('cuda').clone().detach()
        self.regressor.heads.cluster_centers = centers

        # Train the regression head.
        for self.epoch in range(self.options.epochs):
            epoch_start_time = time.time()
            self.run_epoch()
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
        training_dataloader = DataLoader(
            dataset=self.dataset,
            sampler=batch_sampler,
            batch_size=None,
            worker_init_fn=seed_worker,
            generator=self.loader_generator,
            pin_memory=True,
            num_workers=self.num_data_loader_workers,
            persistent_workers=self.num_data_loader_workers > 0,
            # timeout=60 if self.num_data_loader_workers > 0 else 0,
        )

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
                for image_B1HW, image_mask_B1HW, coord_BHW3, gt_pose_B44, gt_pose_inv_B44, intrinsics_B33, intrinsics_inv_B33, semantic_BHW, _ in training_dataloader:
                    # Copy to device.
                    image_B1HW = image_B1HW.to(self.device, non_blocking=True)
                    image_mask_B1HW = image_mask_B1HW.to(self.device, non_blocking=True)
                    coord_BHW3 = coord_BHW3.to(self.device, non_blocking=True)
                    gt_pose_inv_B44 = gt_pose_inv_B44.to(self.device, non_blocking=True)
                    intrinsics_B33 = intrinsics_B33.to(self.device, non_blocking=True)
                    intrinsics_inv_B33 = intrinsics_inv_B33.to(self.device, non_blocking=True)
                    if self.use_semantic_labels:
                        semantic_BHW = semantic_BHW.to(self.device, non_blocking = True)

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
                        'intrinsics_inv': intrinsics_inv
                    }
                    if self.use_semantic_labels:
                        batch_data['feature_labels'] = normalize_shape(semantic_B1HW)

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
        random_indices = torch.randperm(end_index, generator=self.training_generator)

        # Iterate with mini batches.
        for batch_start in range(0, end_index, self.options.batch_size):
            batch_end = batch_start + self.options.batch_size

            # Drop last batch if not full.
            if batch_end > end_index:
                continue

            # Sample indices.
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
            )
            self.iteration += 1

    def training_step(self, features_bC, target_px_b2, target_coord, gt_inv_poses_b34, Ks_b33, invKs_b33, feature_labels, spatial_labels):
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
        if Ks > 1:
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

        # conf judge (only when Ks > 1)
        if Ks > 1:
            loss_euc = self.euclidean_loss.compute(
                target_coord,
                pred_scene_coords_b31.squeeze(),
                None,
                valid_mask_b1
            )
        else:
            loss_euc = torch.zeros((), device=self.device, dtype=torch.float32)

        loss = loss_euc * self.alpha + loss_repro + spatial_labels_loss * 100 + feature_labels_loss * 100

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

