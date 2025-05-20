# Copyright © Niantic, Inc. 2022.

import logging
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from ace_util import one_hot

_logger = logging.getLogger(__name__)

OUTPUT_DIM = 5

class Encoder(nn.Module):
    """
    FCN encoder, used to extract features from the input images.

    The number of output channels is configurable, the default used in the paper is 512.

    Encoder(
        (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (conv4): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (res1_conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (res1_conv2): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (res1_conv3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (res2_conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (res2_conv2): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (res2_conv3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (res2_skip): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
    )
    """

    def __init__(self, out_channels=512):
        super(Encoder, self).__init__()

        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)

        self.res1_conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.res1_conv2 = nn.Conv2d(256, 256, 1, 1, 0)
        self.res1_conv3 = nn.Conv2d(256, 256, 3, 1, 1)

        self.res2_conv1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.res2_conv2 = nn.Conv2d(512, 512, 1, 1, 0)
        self.res2_conv3 = nn.Conv2d(512, self.out_channels, 3, 1, 1)

        self.res2_skip = nn.Conv2d(256, self.out_channels, 1, 1, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        res = F.relu(self.conv4(x))

        x = F.relu(self.res1_conv1(res))
        x = F.relu(self.res1_conv2(x))
        x = F.relu(self.res1_conv3(x))

        res = res + x

        x = F.relu(self.res2_conv1(res))
        x = F.relu(self.res2_conv2(x))
        x = F.relu(self.res2_conv3(x))

        x = self.res2_skip(res) + x

        # x.shape = torch.Size([1, 512, 60, 80])
        return x


class Head(nn.Module):
    """
    MLP network predicting per-pixel scene coordinates given a feature vector. All layers are 1x1 convolutions.

    Head(
        (head_skip): Identity()
        (res3_conv1): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (res3_conv2): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (res3_conv3): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (0c0): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (0c1): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (0c2): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (fc1): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (fc2): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (fc3): Conv2d(512, 4, kernel_size=(1, 1), stride=(1, 1))
    )
    """

    def __init__(self,
                 mean,
                 num_head_blocks,
                 use_homogeneous,
                 feature_clusters,
                 spatial_clusters,
                 homogeneous_min_scale=0.01,
                 homogeneous_max_scale=4.0,
                 in_channels=512):
        super(Head, self).__init__()

        self.use_homogeneous = use_homogeneous
        self.in_channels = in_channels  # Number of encoder features.
        self.head_channels = 512  # Hardcoded.
        self.feature_clusters = feature_clusters
        self.spatial_clusters = spatial_clusters

        # We may need a skip layer if the number of features output by the encoder is different.
        self.head_skip = nn.Identity() if self.in_channels == self.head_channels else nn.Conv2d(self.in_channels,
                                                                                                self.head_channels, 1,
                                                                                                1, 0)

        self.res3_conv1 = nn.Conv2d(self.in_channels, self.head_channels, 1, 1, 0)
        self.res3_conv2 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.res3_conv3 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)

        self.res_blocks = []

        for block in range(num_head_blocks):
            self.res_blocks.append((
                nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
                nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
                nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
            ))

            super(Head, self).add_module(str(block) + 'c0', self.res_blocks[block][0])
            super(Head, self).add_module(str(block) + 'c1', self.res_blocks[block][1])
            super(Head, self).add_module(str(block) + 'c2', self.res_blocks[block][2])

        self.fc1 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.fc2 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)

        if self.use_homogeneous:
            self.fc3 = nn.Conv2d(self.head_channels, OUTPUT_DIM, 1, 1, 0)

            # Use buffers because they need to be saved in the state dict.
            self.register_buffer("max_scale", torch.tensor([homogeneous_max_scale]))
            self.register_buffer("min_scale", torch.tensor([homogeneous_min_scale]))
            self.register_buffer("max_inv_scale", 1. / self.max_scale)
            self.register_buffer("h_beta", math.log(2) / (1. - self.max_inv_scale))
            self.register_buffer("min_inv_scale", 1. / self.min_scale)
        else:
            self.fc3 = nn.Conv2d(self.head_channels, OUTPUT_DIM - 1, 1, 1, 0)

        # Learn scene coordinates relative to a mean coordinate (e.g. center of the scene).
        self.sigmoid = nn.Sigmoid()
        self.register_buffer("mean", mean.clone().detach().view(1, 3, 1, 1))
        self.register_buffer("cluster_centers", torch.zeros((self.spatial_clusters * self.feature_clusters, 3, 1, 1)))

        self.spatial_cluster_fc1 = nn.Conv2d(self.spatial_clusters, self.head_channels, 1, 1, 0)
        self.spatial_cluster_fc2 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.spatial_gamma = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.spatial_beta = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)

        self.feature_cluster_fc1 = nn.Conv2d(self.feature_clusters, self.head_channels, 1, 1, 0)
        self.feature_cluster_fc2 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.feature_gamma = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.feature_beta = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.cond = CondLayer()

        self.feature_regressor = Cluster(feature_clusters, in_channels=self.in_channels)
        self.spatial_regressor = Cluster(spatial_clusters, in_channels=self.in_channels)

    def forward(self, res, feature_labels=None, spatial_labels=None):
        pred_spatial_labels = self.spatial_regressor(res)
        pred_feature_labels = self.feature_regressor(res)
        if feature_labels is None:
            feature_labels = pred_feature_labels
            feature_labels_argmax = torch.argmax(feature_labels, dim=1)
            feature_labels = one_hot(feature_labels_argmax, pred_feature_labels.size()[1])
        if spatial_labels is None:
            spatial_labels = pred_spatial_labels
            spatial_labels_argmax = torch.argmax(spatial_labels, dim=1)
            spatial_labels = one_hot(spatial_labels_argmax, pred_spatial_labels.size()[1])

        x = F.relu(self.res3_conv1(res))
        x = F.relu(self.res3_conv2(x))
        x = F.relu(self.res3_conv3(x))

        res = self.head_skip(res) + x

        out_label = F.relu(self.spatial_cluster_fc1(spatial_labels))
        out_label = F.relu(self.spatial_cluster_fc2(out_label))
        gamma_spatial = F.relu(self.spatial_gamma(out_label))
        beta_spatial = F.relu(self.spatial_beta(out_label))
        res = res + self.cond(res, gamma_spatial, beta_spatial)

        out_label = F.relu(self.feature_cluster_fc1(feature_labels))
        out_label = F.relu(self.feature_cluster_fc2(out_label))
        gamma_feature = F.relu(self.feature_gamma(out_label))
        beta_feature = F.relu(self.feature_beta(out_label))
        res = res + self.cond(res, gamma_feature, beta_feature)

        for res_block in self.res_blocks:
            x = F.relu(res_block[0](res))
            x = F.relu(res_block[1](x))
            x = F.relu(res_block[2](x))
            res = res + x

        sc = F.relu(self.fc1(res))
        sc = F.relu(self.fc2(sc))
        sc = self.fc3(sc)

        conf = sc[:, -1:]
        conf = 1 + conf.clip(max=6.9077).exp()

        if self.use_homogeneous:
            # Dehomogenize coords:
            # Softplus ensures we have a smooth homogeneous parameter with a minimum value = self.max_inv_scale.
            h_slice = F.softplus(sc[:, 3, :, :].unsqueeze(1), beta=self.h_beta.item()) + self.max_inv_scale
            h_slice.clamp_(max=self.min_inv_scale)
            sc = sc[:, :3] / h_slice

        # Add the mean to the predicted coordinates.
        indices1 = torch.argmax(spatial_labels, dim=1)
        indices2 = torch.argmax(feature_labels, dim=1)
        indices = indices2 * self.spatial_clusters + indices1
        mean = self.cluster_centers[indices, :, 0, 0].permute(0, 3, 1, 2)

        sc += mean

        sc = torch.cat((sc, conf), dim=1)

        return sc, pred_spatial_labels, pred_feature_labels, indices.unsqueeze(0)


class Cluster(nn.Module):
    def __init__(self,
                 cluster_number,
                 in_channels=512):
        super(Cluster, self).__init__()
        self.head_channels = 512  # Hardcoded.
        self.cluster_number = cluster_number
        self.in_channels = in_channels  # Number of encoder features.
        self.fc1 = nn.Conv2d(self.in_channels, self.head_channels, 1)
        self.fc2 = nn.Conv2d(self.head_channels, self.head_channels, 1)
        self.fc3 = nn.Conv2d(self.head_channels, self.head_channels, 1)
        self.fc4 = nn.Conv2d(self.head_channels, self.head_channels, 1)
        self.fc5 = nn.Conv2d(self.head_channels, self.head_channels, 1)
        self.fc6 = nn.Conv2d(self.head_channels, self.head_channels, 1)
        self.fc = nn.Conv2d(self.head_channels, self.cluster_number, 1)

    def forward(self, res):
        res = F.relu(self.fc1(res))
        res = F.relu(self.fc2(res))
        res = F.relu(self.fc3(res))
        res = F.relu(self.fc4(res))
        res = F.relu(self.fc5(res))
        res = F.relu(self.fc6(res))

        res = self.fc(res)

        return res


class CondLayer(nn.Module):
    """
    implementation of the element-wise linear modulation layer
    """

    def __init__(self):
        super(CondLayer, self).__init__()
        self.elu = nn.ELU(inplace=True)

    def forward(self, x, gammas, betas):
        return self.elu((gammas * x) + betas)


class Regressor(nn.Module):
    """
    FCN architecture for scene coordinate regression.

    The network predicts a 3d scene coordinates, the output is subsampled by a factor of 8 compared to the input.
    """

    OUTPUT_SUBSAMPLE = 8

    def __init__(self, mean, num_head_blocks, use_homogeneous, feature_clusters, spatial_clusters, num_encoder_features=512):
        """
        Constructor.

        mean: Learn scene coordinates relative to a mean coordinate (e.g. the center of the scene).
        num_head_blocks: How many extra residual blocks to use in the head (one is always used).
        use_homogeneous: Whether to learn homogeneous or 3D coordinates.
        num_encoder_features: Number of channels output of the encoder network.
        """
        super(Regressor, self).__init__()

        self.feature_dim = num_encoder_features
        self.feature_clusters = feature_clusters
        self.spatial_clusters = spatial_clusters

        self.encoder = Encoder(out_channels=self.feature_dim)
        self.heads = Head(mean, num_head_blocks, use_homogeneous, feature_clusters, spatial_clusters, in_channels=self.feature_dim)

    @classmethod
    def create_from_encoder(cls, encoder_state_dict, mean, num_head_blocks, use_homogeneous, feature_clusters, spatial_clusters):
        """
        Create a regressor using a pretrained encoder, loading encoder-specific parameters from the state dict.

        encoder_state_dict: pretrained encoder state dictionary.
        mean: Learn scene coordinates relative to a mean coordinate (e.g. the center of the scene).
        num_head_blocks: How many extra residual blocks to use in the head (one is always used).
        use_homogeneous: Whether to learn homogeneous or 3D coordinates.
        """

        # Number of output channels of the last encoder layer.
        num_encoder_features = encoder_state_dict['res2_conv3.weight'].shape[0]

        # Create a regressor.
        _logger.info(f"Creating Regressor using pretrained encoder with {num_encoder_features} feature size.")
        regressor = cls(mean, num_head_blocks, use_homogeneous, feature_clusters, spatial_clusters, num_encoder_features)

        # Load encoder weights.
        regressor.encoder.load_state_dict(encoder_state_dict)

        # Done.
        return regressor

    @classmethod
    def create_from_state_dict(cls, state_dict, feature_clusters, spatial_clusters):
        """
        Instantiate a regressor from a pretrained state dictionary.

        state_dict: pretrained state dictionary.
        """
        # Mean is zero (will be loaded from the state dict).
        mean = torch.zeros((3,))

        # Count how many head blocks are in the dictionary.
        pattern = re.compile(r"^heads\.\d+c0\.weight$")
        num_head_blocks = sum(1 for k in state_dict.keys() if pattern.match(k))

        # Whether the network uses homogeneous coordinates.
        use_homogeneous = state_dict["heads.fc3.weight"].shape[0] == OUTPUT_DIM

        # Number of output channels of the last encoder layer.
        num_encoder_features = state_dict['encoder.res2_conv3.weight'].shape[0]

        # Create a regressor.
        _logger.info(f"Creating regressor from pretrained state_dict:"
                     f"\n\tNum head blocks: {num_head_blocks}"
                     f"\n\tHomogeneous coordinates: {use_homogeneous}"
                     f"\n\tEncoder feature size: {num_encoder_features}")
        regressor = cls(mean, num_head_blocks, use_homogeneous, feature_clusters, spatial_clusters, num_encoder_features)

        # Load all weights.
        regressor.load_state_dict(state_dict)

        # Done.
        return regressor

    @classmethod
    def create_from_split_state_dict(cls, encoder_state_dict, head_state_dict, feature_clusters, spatial_clusters):
        """
        Instantiate a regressor from a pretrained encoder (scene-agnostic) and a scene-specific head.

        encoder_state_dict: encoder state dictionary
        head_state_dict: scene-specific head state dictionary
        """
        # We simply merge the dictionaries and call the other constructor.
        merged_state_dict = {}

        for k, v in encoder_state_dict.items():
            merged_state_dict[f"encoder.{k}"] = v

        for k, v in head_state_dict.items():
            merged_state_dict[f"heads.{k}"] = v

        return cls.create_from_state_dict(merged_state_dict, feature_clusters, spatial_clusters)

    def load_encoder(self, encoder_dict_file):
        """
        Load weights into the encoder network.
        """
        self.encoder.load_state_dict(torch.load(encoder_dict_file))

    def get_features(self, inputs):
        return self.encoder(inputs)

    def get_scene_coordinates(self, features, feature_labels=None, spatial_labels=None):
        return self.heads(features, feature_labels, spatial_labels)

    def get_cluster_labels(self, features):
        return self.heads.clusters(features)

    def set_freeze(self, modules):
        for module in modules:
            try:
                for n, param in module.named_parameters():
                    param.requires_grad = False
            except AttributeError:
                # module is directly a parameter
                module.requires_grad = False

    def forward(self, inputs):
        """
        Forward pass.
        """
        features = self.get_features(inputs)
        return self.get_scene_coordinates(features)
