import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from landmark_detection.face_alignment.face_alignment import FaceAlignment
from torchvision import transforms

import kornia
from kornia.losses import total_variation
from kornia.utils import create_meshgrid

from prnet.prnet import PRNet
import render

from PIL import Image


class LandmarkExtractor(nn.Module):
    def __init__(self, device, face_landmark_detector, img_size):
        super(LandmarkExtractor, self).__init__()
        self.device = device
        self.face_align = face_landmark_detector
        self.img_size_width = img_size[1]
        self.img_size_height = img_size[0]

    def forward(self, img_batch):
        if isinstance(self.face_align, FaceAlignment):
            with torch.no_grad():
                points = self.face_align.get_landmarks_from_batch(img_batch * 255)
            single_face_points = [landmarks[:68] for landmarks in points]
            preds = torch.tensor(single_face_points, device=self.device)
        else:
            with torch.no_grad():
                preds = self.face_align(img_batch)[0].view(img_batch.shape[0], -1, 2)
                preds[..., 0] = preds[..., 0] * self.img_size_height
                preds[..., 1] = preds[..., 1] * self.img_size_width
                preds = preds.type(torch.int)
        return preds


class FaceXZooProjector(nn.Module):
    def __init__(self, device, img_size, patch_size):
        super(FaceXZooProjector, self).__init__()
        self.prn = PRN('../prnet/prnet.pth', device, patch_size[0])

        self.img_size_width = img_size[1]
        self.img_size_height = img_size[0]
        self.patch_size_width = patch_size[1]
        self.patch_size_height = patch_size[0]

        self.device = device
        self.uv_mask_src = transforms.ToTensor()(Image.open('../prnet/new_uv.png').convert('L')).to(device).unsqueeze(0)

        image_info = torch.nonzero(self.uv_mask_src, as_tuple=False)
        left, _ = torch.min(image_info[:, 3], dim=0)
        right, _ = torch.max(image_info[:, 3], dim=0)
        self.mask_half_width = ((right - left) / 2) + 5
        top, _ = torch.min(image_info[:, 2], dim=0)
        bottom, _ = torch.max(image_info[:, 2], dim=0)
        self.mask_half_height = ((bottom - top) / 2)
        self.patch_bbox = self.get_bbox(self.uv_mask_src)

        self.uv_face_src = transforms.ToTensor()(Image.open('../prnet/uv_face_mask.png').convert('L')).to(
            device).unsqueeze(0)
        self.triangles = torch.from_numpy(np.loadtxt('../prnet/triangles.txt').astype(np.int64)).T.to(device)
        self.minangle = -5 / 180 * math.pi
        self.maxangle = 5 / 180 * math.pi
        self.min_trans_x = -0.05
        self.max_trans_x = 0.05
        self.min_trans_y = -0.05
        self.max_trans_y = 0.05
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.05

    def forward(self, img_batch, landmarks, adv_patch, uv_mask_src=None, do_aug=False, is_3d=False):
        pos_orig, vertices_orig = self.get_vertices(img_batch, landmarks)
        texture_img = kornia.geometry.remap(img_batch, map_x=pos_orig[:, 0], map_y=pos_orig[:, 1],
                                            mode='nearest') * self.uv_face_src

        adv_patch = adv_patch.expand(img_batch.shape[0], -1, -1, -1)
        if not is_3d:
            adv_patch_other = self.align_patch(adv_patch, landmarks)
            texture_patch = kornia.geometry.remap(adv_patch_other, map_x=pos_orig[:, 0], map_y=pos_orig[:, 1],
                                                  mode='nearest') * self.uv_face_src
            uv_mask_src = torch.where(texture_patch.sum(dim=1, keepdim=True) != 0, torch.ones(1, device=self.device),
                                      torch.zeros(1, device=self.device))
        else:
            uv_mask_src = uv_mask_src.repeat(adv_patch.shape[0], 1, 1, 1)
            if adv_patch.shape[2] != 256:
                adv_patch = F.interpolate(adv_patch, (256, 256))
                uv_mask_src = F.interpolate(uv_mask_src, (256, 256))
            texture_patch = adv_patch

        if do_aug:
            texture_patch, uv_mask_src = self.augment_patch(texture_patch, uv_mask_src)

        new_texture = texture_patch * uv_mask_src + texture_img * (1 - uv_mask_src)

        new_colors = self.prn.get_colors_from_texture(new_texture)

        face_mask, new_image = render.render_cy_pt(vertices_orig,
                                                   new_colors,
                                                   self.triangles,
                                                   img_batch.shape[0],
                                                   self.img_size_height,
                                                   self.img_size_width,
                                                   self.device)
        face_mask = torch.where(torch.floor(face_mask) > 0,
                                torch.ones(1, device=self.device),
                                torch.zeros(1, device=self.device))
        new_image = img_batch * (1 - face_mask) + (new_image * face_mask)
        new_image.data.clamp_(0, 1)

        return new_image

    def align_patch(self, adv_patch, landmarks):
        batch_size = landmarks.shape[0]
        src_pts = self.patch_bbox.repeat(batch_size, 1, 1)

        landmarks = landmarks.type(torch.float32)
        max_side_dist = torch.maximum(landmarks[:, 33, 0]-landmarks[:, 2, 0], landmarks[:, 14, 0]-landmarks[:, 33, 0])
        max_side_dist = torch.where(max_side_dist < self.mask_half_width, self.mask_half_width, max_side_dist)

        left_top = torch.stack((landmarks[:, 33, 0]-max_side_dist, landmarks[:, 62, 1]-self.mask_half_height), dim=-1)
        right_top = torch.stack((landmarks[:, 33, 0]+max_side_dist, landmarks[:, 62, 1]-self.mask_half_height), dim=-1)
        left_bottom = torch.stack((landmarks[:, 33, 0]-max_side_dist,  landmarks[:, 62, 1]+self.mask_half_height), dim=-1)
        right_bottom = torch.stack((landmarks[:, 33, 0]+max_side_dist, landmarks[:, 62, 1]+self.mask_half_height), dim=-1)
        dst_pts = torch.stack([left_top, right_top, left_bottom, right_bottom], dim=1)

        tform = kornia.find_homography_dlt(src_pts, dst_pts)
        cropped_image = kornia.geometry.warp_perspective(adv_patch, tform, dsize=(self.img_size_width, self.img_size_height), mode='nearest')

        grid = create_meshgrid(112, 112, False, device=self.device).repeat(batch_size, 1, 1, 1)

        for i in range(batch_size):
            bbox_info = self.get_bbox(cropped_image[i:i+1])
            left_top = bbox_info[:, 0]
            right_top = bbox_info[:, 1]
            x_center = (right_top[:, 0] - left_top[:, 0]) / 2
            target_y = torch.mean(torch.stack([landmarks[i, 0, 1], landmarks[i, 0, 1]]))
            max_y_left = torch.clamp_min(-(target_y - left_top[:, 1]), 0)
            start_idx_left = min(int(left_top[0, 0].item()), self.img_size_width)
            end_idx_left = min(int(start_idx_left + x_center), self.img_size_width)
            offset = torch.zeros_like(grid[i, :, start_idx_left:end_idx_left, 1])
            dropoff = 0.97
            for j in range(offset.shape[1]):
                offset[:, j] = (max_y_left - ((j*max_y_left)/offset.shape[1])) * dropoff

            grid[i, :, start_idx_left:end_idx_left, 1] = grid[i, :, start_idx_left:end_idx_left, 1] + offset

            target_y = torch.mean(torch.stack([landmarks[i, 16, 1], landmarks[i, 16, 1]]))
            max_y_right = torch.clamp_min(-(target_y - right_top[:, 1]), 0)
            end_idx_right = min(int(right_top[0, 0].item()), self.img_size_width) + 1
            start_idx_right = min(int(end_idx_right - x_center), self.img_size_width)
            offset = torch.zeros_like(grid[i, :, start_idx_right:end_idx_right, 1])
            for idx, col in enumerate(reversed(range(offset.shape[1]))):
                offset[:, col] = (max_y_right - ((idx*max_y_right)/offset.shape[1])) * dropoff
            grid[i, :, start_idx_right:end_idx_right, 1] = grid[i, :, start_idx_right:end_idx_right, 1] + offset

        cropped_image = kornia.remap(cropped_image, map_x=grid[..., 0], map_y=grid[..., 1], mode='nearest')
        return cropped_image

    def get_bbox(self, adv_patch):
        image_info = torch.nonzero(adv_patch, as_tuple=False).unsqueeze(0)
        left, _ = torch.min(image_info[:, :, 3], dim=1)
        right, _ = torch.max(image_info[:, :, 3], dim=1)
        top, _ = torch.min(image_info[:, :, 2], dim=1)
        bottom, _ = torch.max(image_info[:, :, 2], dim=1)
        width = right - left
        height = bottom - top
        # crop image
        center = torch.stack([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0], dim=-1)
        left_top = torch.stack((center[:, 0] - (width / 2), center[:, 1] - (height / 2)), dim=-1)
        right_top = torch.stack((center[:, 0] + (width / 2), center[:, 1] - (height / 2)), dim=-1)
        left_bottom = torch.stack((center[:, 0] - (width / 2), center[:, 1] + (height / 2)), dim=-1)
        right_bottom = torch.stack((center[:, 0] + (width / 2), center[:, 1] + (height / 2)), dim=-1)
        src_pts = torch.stack([left_top, right_top, left_bottom, right_bottom], dim=1)
        return src_pts

    def get_vertices(self, image, face_lms):
        pos = self.prn.process(image, face_lms)
        vertices = self.prn.get_vertices(pos)
        return pos, vertices

    def augment_patch(self, adv_patch, uv_mask_src):
        contrast = self.get_random_tensor(adv_patch, self.min_contrast, self.max_contrast) * uv_mask_src
        brightness = self.get_random_tensor(adv_patch, self.min_brightness, self.max_brightness) * uv_mask_src
        noise = torch.empty(adv_patch.shape, device=self.device).uniform_(-1, 1) * self.noise_factor * uv_mask_src
        adv_patch = adv_patch * contrast + brightness + noise
        adv_patch.data.clamp_(0.000001, 0.999999)
        merged = torch.cat([adv_patch, uv_mask_src], dim=1)
        merged_aug = self.apply_random_grid_sample(merged)
        adv_patch = merged_aug[:, :3]
        uv_mask_src = merged_aug[:, 3:]
        return adv_patch, uv_mask_src

    def get_random_tensor(self, adv_patch, min_val, max_val):
        t = torch.empty(adv_patch.shape[0], device=self.device).uniform_(min_val, max_val)
        t = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        t = t.expand(-1, adv_patch.size(-3), adv_patch.size(-2), adv_patch.size(-1))
        return t

    def apply_random_grid_sample(self, face_mask):
        theta = torch.zeros((face_mask.shape[0], 2, 3), dtype=torch.float, device=self.device)
        rand_angle = torch.empty(face_mask.shape[0], device=self.device).uniform_(self.minangle, self.maxangle)
        theta[:, 0, 0] = torch.cos(rand_angle)
        theta[:, 0, 1] = -torch.sin(rand_angle)
        theta[:, 1, 1] = torch.cos(rand_angle)
        theta[:, 1, 0] = torch.sin(rand_angle)
        theta[:, 0, 2].uniform_(self.min_trans_x, self.max_trans_x)  # move x
        theta[:, 1, 2].uniform_(self.min_trans_y, self.max_trans_y)  # move y
        grid = F.affine_grid(theta, list(face_mask.size()))
        augmented = F.grid_sample(face_mask, grid, padding_mode='reflection')
        return augmented


class PRN:
    """Process of PRNet.
    based on:
    https://github.com/YadiraF/PRNet/blob/master/api.py
    """
    def __init__(self, model_path, device):
        self.resolution = 256
        self.MaxPos = self.resolution * 1.1
        self.face_ind = np.loadtxt('../prnet/face_ind.txt').astype(np.int32)
        self.triangles = np.loadtxt('../prnet/triangles.txt').astype(np.int32)
        self.net = PRNet(3, 3)
        self.net.load_state_dict(torch.load(model_path))
        self.device = device
        self.net.to(device).eval()

    def get_bbox_annot(self, image_info):
        left, _ = torch.min(image_info[..., 0], dim=1)
        right, _ = torch.max(image_info[..., 0], dim=1)
        top, _ = torch.min(image_info[..., 1], dim=1)
        bottom, _ = torch.max(image_info[..., 1], dim=1)
        return left, right, top, bottom

    def preprocess(self, img_batch, image_info):
        left, right, top, bottom = self.get_bbox_annot(image_info)
        center = torch.stack([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0], dim=-1)

        old_size = (right - left + bottom - top) / 2
        size = (old_size * 1.5).type(torch.int32)

        # crop image
        left_top = torch.stack((center[:, 0] - (size / 2), center[:, 1] - (size / 2)), dim=-1)
        right_top = torch.stack((center[:, 0] + (size / 2), center[:, 1] - (size / 2)), dim=-1)
        left_bottom = torch.stack((center[:, 0] - (size / 2), center[:, 1] + (size / 2)), dim=-1)
        right_bottom = torch.stack((center[:, 0] + (size / 2), center[:, 1] + (size / 2)), dim=-1)
        src_pts = torch.stack([left_top, right_top, left_bottom, right_bottom], dim=1)
        dst_pts = torch.tensor([[0, 0],
                                [self.resolution - 1, 0],
                                [0, self.resolution - 1],
                                [self.resolution - 1, self.resolution - 1]],
                               dtype=torch.float32, device=self.device).repeat(src_pts.shape[0], 1, 1)

        tform = kornia.get_perspective_transform(src_pts, dst_pts)
        cropped_image = kornia.geometry.warp_perspective(img_batch, tform, dsize=(self.resolution, self.resolution))

        return cropped_image, tform

    def process(self, img_batch, image_info):
        cropped_image, tform = self.preprocess(img_batch, image_info)

        cropped_pos = self.net(cropped_image)
        cropped_vertices = (cropped_pos * self.MaxPos).view(cropped_pos.shape[0], 3, -1)

        z = cropped_vertices[:, 2:3, :].clone() / tform[:, :1, :1]
        cropped_vertices[:, 2, :] = 1

        vertices = torch.bmm(torch.linalg.inv(tform), cropped_vertices)
        vertices = torch.cat((vertices[:, :2, :], z), dim=1)

        pos = vertices.reshape(vertices.shape[0], 3, self.resolution, self.resolution)
        return pos

    def get_vertices(self, pos):
        all_vertices = pos.view(pos.shape[0], 3, -1)
        vertices = all_vertices[..., self.face_ind]
        return vertices

    def get_colors_from_texture(self, texture):
        all_colors = texture.view(texture.shape[0], 3, -1)
        colors = all_colors[..., self.face_ind]
        return colors


class TotalVariation(nn.Module):
    def __init__(self, device) -> None:
        super(TotalVariation, self).__init__()
        self.device = device
        self.uv_mask_src = transforms.ToTensor()(Image.open('../prnet/new_uv.png').convert('L')).to(device).squeeze()
        self.number_of_pixels = torch.count_nonzero(self.uv_mask_src)
        self.save_grads = torch.zeros_like(self.uv_mask_src)

    def forward(self, adv_patch):
        tv_patch = adv_patch * self.uv_mask_src
        loss = total_variation(tv_patch) / self.number_of_pixels
        return loss

    def zero_grads(self, grads):
        self.save_grads = torch.where(grads.sum(dim=1).sum(dim=0).unsqueeze(0).unsqueeze(0) != 0,
                                      torch.ones(1, device=self.device),
                                      self.save_grads)
