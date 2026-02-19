import torch


class SemanticDPPrototypeManager:
    """Streaming DP-style prototype manager for sparse semantic samples."""

    def __init__(self, options, device):
        self.options = options
        self.device = device
        self.max_proto = int(options.dp_max_proto_per_class)
        self.ema_beta = float(options.dp_proto_ema_beta)
        self.default_depth = float(options.dp_default_depth)
        self.use_depth_prior = bool(options.dp_use_depth_prior)
        self.class_centers = {}
        self.class_counts = {}

    @staticmethod
    def _homogeneous_pixels(target_px):
        ones = torch.ones((target_px.shape[0], 1), device=target_px.device, dtype=target_px.dtype)
        return torch.cat([target_px, ones], dim=1)

    def _pseudo_world_points(self, target_px, intrinsics_inv, gt_pose_inv, depth_values):
        pix_h = self._homogeneous_pixels(target_px).unsqueeze(-1)  # Nx3x1
        ray_cam = torch.bmm(intrinsics_inv.float(), pix_h).squeeze(-1)
        ray_cam = torch.nn.functional.normalize(ray_cam, dim=1)

        R_cw = gt_pose_inv[:, :3, :3].float()
        t_cw = gt_pose_inv[:, :3, 3].float()
        R_wc = R_cw.transpose(1, 2)
        cam_center = -torch.bmm(R_wc, t_cw.unsqueeze(-1)).squeeze(-1)
        ray_world = torch.bmm(R_wc, ray_cam.unsqueeze(-1)).squeeze(-1)
        ray_world = torch.nn.functional.normalize(ray_world, dim=1)

        if self.use_depth_prior:
            depth = depth_values.view(-1).float()
            valid = depth > 0
            depth = torch.where(valid, depth, torch.full_like(depth, self.default_depth))
        else:
            depth = torch.full((target_px.shape[0],), self.default_depth, device=target_px.device, dtype=torch.float32)

        world_points = cam_center + depth.unsqueeze(-1) * ray_world
        return world_points, depth

    def _tau_for_points(self, depth):
        if self.options.dp_tau_mode == 'noise_model':
            return self.options.dp_tau * (1.0 + self.options.dp_tau_scale * depth)
        return torch.full_like(depth, float(self.options.dp_tau))

    def assign(self, semantic_labels, target_px, intrinsics_inv, gt_pose_inv, depth_values):
        labels = semantic_labels.view(-1).to(torch.int64)
        world_points, depth = self._pseudo_world_points(target_px, intrinsics_inv, gt_pose_inv, depth_values)
        tau_values = self._tau_for_points(depth)

        output = torch.zeros_like(labels)
        unique_classes = torch.unique(labels)

        for cls in unique_classes.tolist():
            cls_mask = labels == cls
            cls_indices = torch.nonzero(cls_mask, as_tuple=False).view(-1)
            if cls_indices.numel() == 0:
                continue

            points = world_points[cls_indices]
            cls_tau = tau_values[cls_indices]

            if cls not in self.class_centers:
                self.class_centers[cls] = torch.empty((0, 3), device=self.device, dtype=torch.float32)
                self.class_counts[cls] = torch.empty((0,), device=self.device, dtype=torch.float32)

            centers = self.class_centers[cls]
            counts = self.class_counts[cls]
            assignments_local = []

            for point, tau_i in zip(points, cls_tau):
                point = point.to(self.device)
                if centers.shape[0] == 0:
                    centers = point.unsqueeze(0)
                    counts = torch.ones((1,), device=self.device, dtype=torch.float32)
                    assign_idx = 0
                else:
                    dists = torch.norm(centers - point.unsqueeze(0), dim=1)
                    min_dist, min_idx = torch.min(dists, dim=0)

                    if min_dist > tau_i and centers.shape[0] < self.max_proto:
                        centers = torch.cat([centers, point.unsqueeze(0)], dim=0)
                        counts = torch.cat([counts, torch.ones((1,), device=self.device, dtype=torch.float32)], dim=0)
                        assign_idx = centers.shape[0] - 1
                    else:
                        assign_idx = int(min_idx.item())
                        centers[assign_idx] = self.ema_beta * centers[assign_idx] + (1.0 - self.ema_beta) * point
                        counts[assign_idx] = counts[assign_idx] + 1.0

                assignments_local.append(assign_idx)

            self.class_centers[cls] = centers
            self.class_counts[cls] = counts
            assignments_local = torch.tensor(assignments_local, device=labels.device, dtype=torch.int64)
            output[cls_indices] = labels[cls_indices] * self.max_proto + assignments_local

        return output.view(-1, 1).to(torch.float32)

    def summary(self):
        parts = []
        for cls in sorted(self.class_centers.keys()):
            parts.append(f"c{cls}:{self.class_centers[cls].shape[0]}")
        return " ".join(parts) if parts else "no prototypes"
