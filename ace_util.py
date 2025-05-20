# Copyright © Niantic, Inc. 2022.
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F


def get_pixel_grid(subsampling_factor):
    """
    Generate target pixel positions according to a subsampling factor, assuming prediction at center pixel.
    """
    pix_range = torch.arange(np.ceil(5000 / subsampling_factor), dtype=torch.float32)
    yy, xx = torch.meshgrid(pix_range, pix_range, indexing='ij')
    return subsampling_factor * (torch.stack([xx, yy]) + 0.5)


def to_homogeneous(input_tensor, dim=1):
    """
    Converts tensor to homogeneous coordinates by adding ones to the specified dimension
    """
    ones = torch.ones_like(input_tensor.select(dim, 0).unsqueeze(dim))
    output = torch.cat([input_tensor, ones], dim=dim)
    return output


def get_coord(depth, pose, intrinsics_color_inv):
    """Generate the ground truth scene coordinates from depth and pose.
    """
    img_height, img_width = depth.shape[0], depth.shape[1]
    mask = np.ones_like(depth)
    mask[depth == 0] = 0
    mask = np.reshape(mask, (img_height, img_width, 1))
    x = np.linspace(0, img_width - 1, img_width)
    y = np.linspace(0, img_height - 1, img_height)
    xx, yy = np.meshgrid(x, y)

    # xx: [[  0.   1.   2. ... 637. 638. 639.]]
    # yy: [[  0.   0.   0. ... 479. 479. 479.]]
    xx = np.reshape(xx, (1, -1))
    yy = np.reshape(yy, (1, -1))
    ones = np.ones_like(xx)
    pcoord = np.concatenate((xx, yy, ones), axis=0)
    depth = np.reshape(depth, (1, img_height * img_width))
    ccoord = np.dot(intrinsics_color_inv, pcoord) * depth
    ccoord = np.concatenate((ccoord, ones), axis=0)
    scoord = np.dot(pose, ccoord)
    scoord = np.swapaxes(scoord, 0, 1)
    scoord = scoord[:, 0:3]
    scoord = np.reshape(scoord, (img_height, img_width, 3))
    scoord *= mask
    mask = np.reshape(mask, (img_height, img_width))

    return scoord, mask

def to_tensor(coord_img, mask):
    coord_img = coord_img / 1000.
    coord_img = torch.from_numpy(coord_img).float()
    mask = torch.from_numpy(mask).float()

    return coord_img, mask

def calculate_magnitude(image):
    kernel = torch.tensor([[[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]]], dtype=torch.float16).to('cuda')

    kernel = kernel.view(1, 1, 3, 3)

    # transform = torch.nn.functional.affine_grid(torch.eye(2, 3), image.size, align_corners=True)
    # image_tensor = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0)

    output = F.conv2d(image, kernel, stride=1, padding=1)

    output = torch.abs(output).squeeze()
    output = (output - output.min()) / (output.max() - output.min())
    # output_numpy = torch.abs(output).squeeze().cpu().detach().numpy()
    # output_numpy = (output_numpy - output_numpy.min()) / (output_numpy.max() - output_numpy.min())

    import matplotlib.pyplot as plt

    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.imshow(image[0][0], cmap='gray')
    # plt.title('Original Image')

    # plt.subplot(1, 2, 2)
    # plt.imshow(output_numpy, cmap='gray')
    # plt.title('After Convolution')
    #
    # plt.show()
    return output

def calculate_euclidean(pred, truth, mask, conf):
    H, W, C = pred.shape
    mask = TF.resize(mask, [H, W], interpolation=TF.InterpolationMode.NEAREST)[0, 0]
    pred = pred * mask.unsqueeze(-1)
    truth = truth * mask.unsqueeze(-1)

    dist = torch.norm((pred - truth), dim=-1)
    dist_numpy = dist.cpu().detach().numpy()
    dist_numpy = (dist_numpy - dist_numpy.min()) / (dist_numpy.max() - dist_numpy.min())

    truth_numpy = truth[:, :, 2].cpu().detach().numpy()
    truth_numpy = (truth_numpy - truth_numpy.min()) / (truth_numpy.max() - truth_numpy.min())
    pred_numpy = pred[:, :, 2].cpu().detach().numpy()
    pred_numpy = (pred_numpy - pred_numpy.min()) / (pred_numpy.max() - pred_numpy.min())

    conf_numpy = conf.cpu().detach().numpy()
    conf_numpy = (conf_numpy - conf_numpy.min()) / (conf_numpy.max() - conf_numpy.min())

    cv2.imshow('test', np.concatenate((dist_numpy, truth_numpy, pred_numpy, conf_numpy), axis=1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return dist

def one_hot(x, N=25):
    one_hot = torch.FloatTensor(x.size(0), N, x.size(1),
                                x.size(2)).zero_().to(x.device)
    one_hot = one_hot.scatter_(1, x.unsqueeze(1), 1)
    return one_hot
