import PIL.Image
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import math
import numpy as np
import torch.nn.functional as F
from landmark_detection.face_alignment.face_alignment import FaceAlignment
from torchvision import transforms

import kornia
from kornia.geometry.homography import find_homography_dlt
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
        self.mask_half_width = torch.tensor((right - left) / 2, device=device, dtype=torch.float32) + 5
        top, _ = torch.min(image_info[:, 2], dim=0)
        bottom, _ = torch.max(image_info[:, 2], dim=0)
        self.mask_half_height = torch.tensor((bottom - top) / 2, device=device, dtype=torch.float32)
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
        self.noise_factor = 0.1

    def forward(self, img_batch, landmarks, adv_patch, uv_mask_src=None, do_aug=False, is_3d=False):
        # transforms.ToPILImage()(img_batch[0].detach().cpu()).show()
        pos_orig, vertices_orig = self.get_vertices(img_batch, landmarks)
        texture_img = kornia.geometry.remap(img_batch, map_x=pos_orig[:, 0], map_y=pos_orig[:, 1],
                                            mode='nearest') * self.uv_face_src
        # transforms.ToPILImage()(texture_img[0].detach().cpu()).show()

        adv_patch = adv_patch.expand(img_batch.shape[0], -1, -1, -1)
        # transforms.ToPILImage()(adv_patch[0].detach().cpu()).show()
        if not is_3d:
            adv_patch_other = self.align_patch(adv_patch, landmarks)
            for i in range(adv_patch_other.shape[0]):
               transforms.ToPILImage()(torch.where(adv_patch_other[i].sum(dim=0) != 0, adv_patch_other[i], img_batch[i])).show()
            # transforms.ToPILImage()(adv_patch_other[0].detach().cpu()).show()
            texture_patch = kornia.geometry.remap(adv_patch_other, map_x=pos_orig[:, 0], map_y=pos_orig[:, 1],
                                                  mode='nearest') * self.uv_face_src
            # transforms.ToPILImage()(texture_patch[0].detach().cpu()).show()
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
            # transforms.ToPILImage()(texture_patch[0].detach().cpu()).show()

        new_texture = texture_patch * uv_mask_src + texture_img * (1 - uv_mask_src)
        # transforms.ToPILImage()(new_texture[0].detach().cpu()).show()

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

        for i in range(new_image.shape[0]):
            transforms.ToPILImage()(new_image[i]).show()
        return new_image

    def align_patch_old(self, adv_patch, landmarks):
        # image_info = torch.nonzero(adv_patch, as_tuple=False)
        # left_upper_idx = torch.min(image_info[:, 3], dim=0)[1]
        # right_idx = torch.topk(image_info[:, 3], k=4, dim=0)[1]
        # top_max_idx = torch.min(image_info[:, 2], dim=0)[1]
        # bottom_idx = torch.max(image_info[:, 2], dim=0)[1]

        src_pts = torch.tensor([[114, 83],  # left nose
                                [50,  100],  # left upper edge
                                [50,  156],  # left lower edge
                                [59,  161],  # 1/5 left chin
                                [70,  165],  # 2/5 left chin
                                [85,  169],  # 3/5 left chin
                                # [96,  171],  # 4/5 left chin
                                [128, 173],  # middle chin
                                # [160, 171],  # 4/5 right chin
                                [171, 169],  # 3/5 right chin
                                [186, 165],  # 2/5 right chin
                                [197, 161],  # 1/5 right chin
                                [205, 156],  # right lower edge
                                [205, 100],  # right upper edge
                                [142, 83]  # right noise
        ], dtype=torch.float32, device=self.device).repeat(landmarks.shape[0], 1, 1)
        landmarks = landmarks.type(torch.float32)
        dst_pts = torch.stack([landmarks[:, 31],
                               landmarks[:, 0],
                               landmarks[:, 1],
                               landmarks[:, 4],
                               landmarks[:, 5],
                               landmarks[:, 6],
                               landmarks[:, 8],
                               landmarks[:, 10],
                               landmarks[:, 11],
                               landmarks[:, 12],
                               landmarks[:, 15],
                               landmarks[:, 16],
                               landmarks[:, 35],
                               ], dim=1)
        kernel_weights, affine_weights = kornia.geometry.get_tps_transform(dst_pts, src_pts)
        cropped_image = kornia.warp_image_tps(adv_patch, src_pts, kernel_weights, affine_weights)
        # tform = kornia.geometry.find_homography_dlt(src_pts, dst_pts)
        # resolution = 112
        # cropped_image = kornia.geometry.warp_perspective(adv_patch, tform, dsize=(resolution, resolution),
        #                                                mode='nearest')
        # transforms.ToPILImage()(cropped_image[0]).show()
        return cropped_image

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
        # transforms.ToPILImage()(cropped_image[0]).show()

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
        # transforms.ToPILImage()(cropped_image[0]).show()
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
        """Get vertices

        Args:
            face_lms: face landmarks.
            image:[0, 255]
        """
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
        # transforms.ToPILImage()(adv_patch[0]).show()
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

    def __init__(self, model_path, device, resolution):
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

    def preprocess(self, img_batch, image_info, resolution=None):
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
        cropped_image, tform = self.preprocess(img_batch, image_info, self.resolution)

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


class NormalizeToArcFace(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.five_point_lm_indices = [41, 47, 34, 49, 55]
        self.arcface_src = torch.tensor([[[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                                          [41.5493, 92.3655], [70.7299, 92.2041]]]).to(device)

    def forward(self, img_batch, landmarks):
        transforms.ToPILImage()(img_batch[0].detach().cpu()).show()
        dst_pts = landmarks[:, self.five_point_lm_indices].type(torch.float32)
        tform = kornia.geometry.find_homography_dlt(self.arcface_src, dst_pts)
        normalized = kornia.geometry.transform.warp_affine(img_batch, tform[:, 0:2, :], (112, 112))
        transforms.ToPILImage()(normalized[0].detach().cpu()).show()
        return normalized


class PatchTransformer(nn.Module):
    def __init__(self, device, img_size, patch_size):
        super(PatchTransformer, self).__init__()
        self.device = device
        self.img_size_width = img_size[1]
        self.img_size_height = img_size[0]

        self.patch_size_width = patch_size[1]
        self.patch_size_height = patch_size[0]

    def forward(self, adv_patch, lab_batch, preds):
        pad_height = int((self.img_size_height - self.patch_size_height) / 2)
        pad_width = int((self.img_size_width - self.patch_size_width) / 2)
        adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)
        mypad = nn.ConstantPad2d((pad_width, pad_width, pad_height, pad_height), 0)
        adv_batch = mypad(adv_batch)
        # transforms.ToPILImage()(adv_batch[0][0]).show()

        batch_size = lab_batch.size()[:2]
        target_x = lab_batch[..., 1].view(np.prod(batch_size))
        target_y = lab_batch[..., 2].view(np.prod(batch_size))
        s = adv_batch.size()
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])
        tx = (-target_x + 0.5) * 2  # [0,1] to [-1,1]
        ty = (-target_y + 0.5) * 2  # [0,1] to [-1,1]
        ty = ty + 0.1
        anglesize = (lab_batch.size(0) * lab_batch.size(1))

        scaled_width = lab_batch[:, :, 3] * self.img_size_width
        scaled_height = lab_batch[:, :, 4] * self.img_size_height
        # target_size_x = torch.sqrt(((scaled_width) ** 2) + ((scaled_height) ** 2))

        scale_x = self.patch_size_width / scaled_width
        scale_x = scale_x.view(batch_size[0] * batch_size[1])
        scale_y = self.patch_size_height / scaled_height
        scale_y = scale_y.view(batch_size[0] * batch_size[1])

        theta = torch.zeros((anglesize, 2, 3), device=self.device)
        theta[:, 0, 0] = scale_x
        theta[:, 0, 1] = 0
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = 0
        theta[:, 1, 1] = scale_y
        theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch, grid, padding_mode='border')
        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])
        adv_batch_t = torch.clamp(adv_batch_t, 0, 0.999999)

        # transforms.ToPILImage()(adv_batch_t[0][0]).show()
        return adv_batch_t

    # def forward(self, adv_patch, lab_batch, preds):
    #     pad_height = (self.img_size_height - self.patch_size_height) / 2
    #     pad_width = (self.img_size_width - self.patch_size_width) / 2
    #     adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)
    #     mypad = nn.ConstantPad2d((int(pad_width), int(pad_width), int(pad_height), int(pad_height)), 0)
    #     adv_batch = mypad(adv_batch)
    #     transforms.ToPILImage()(adv_batch[0][0]).show()
    #
    #     batch_size = lab_batch.size()[:2]
    #
    #     scaled_width = lab_batch[:, :, 3] * self.img_size_width
    #     scaled_height = lab_batch[:, :, 4] * self.img_size_height
    #
    #     target_size = torch.sqrt(((scaled_width*0.9) ** 2) + ((scaled_height*0.9) ** 2))
    #     target_x = lab_batch[:, :, 1].view(np.prod(batch_size))
    #     target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
    #
    #     # targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))
    #     # targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))
    #
    #     # target_y = target_y - 0.05
    #     scale_x = target_size / self.patch_size_width
    #     scale_x = scale_x.view(batch_size[0]*batch_size[1])
    #     scale_y = target_size / self.patch_size_height
    #     scale_y = scale_y.view(batch_size[0]*batch_size[1])
    #
    #     s = adv_batch.size()
    #     adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])
    #
    #     anglesize = (lab_batch.size(0) * lab_batch.size(1))
    #     # point1 = preds[:, 2, :]
    #     # point2 = preds[:, 14, :]
    #     # point_insct = np.concatenate([preds[:,2,0], preds[:, 14, 1]])
    #     #
    #     angle = torch.zeros(anglesize)
    #     tx = (-target_x+0.5)*2
    #     # tx = target_x
    #     ty = (-target_y+0.5)*2
    #     # ty = target_y
    #     sin = torch.sin(angle)
    #     cos = torch.cos(angle)
    #
    #     theta = torch.zeros((anglesize, 2, 3), device=self.device)
    #     theta[:, 0, 0] = cos/scale_x
    #     theta[:, 0, 1] = sin/scale_y
    #     theta[:, 0, 2] = tx*cos/scale_x+ty*sin/scale_y
    #     theta[:, 1, 0] = -sin/scale_y
    #     theta[:, 1, 1] = cos/scale_x
    #     theta[:, 1, 2] = -tx*sin/scale_x+ty*cos/scale_y
    #
    #     grid = F.affine_grid(theta, adv_batch.shape)
    #     adv_batch_t = F.grid_sample(adv_batch, grid)
    #     adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])
    #     adv_batch_t = torch.clamp(adv_batch_t, 0, 0.999999)
    #
    #     transforms.ToPILImage()(adv_batch_t[0][0]).show()
    #     return adv_batch_t


class LandmarksApplier(nn.Module):
    def __init__(self, indices):
        super(LandmarksApplier, self).__init__()
        self.indices = indices

    def forward(self, img_batch, points):
        img = transforms.ToPILImage()(img_batch[0])
        for detection in points:
            plt.imshow(img)
            detection[36, 1] = detection[29, 1]
            detection[45, 1] = detection[29, 1]
            plt.fill(detection[self.indices, 0], detection[self.indices, 1], color=(0.4, 1, 1), edgecolor='white')
            plt.scatter(detection[self.indices, 0], detection[self.indices, 1], 2)
            plt.show()
        print('x')


class Projector(nn.Module):
    def __init__(self, parabola_rate, rotation_angle, patch_size, batch_size):
        super(Projector, self).__init__()
        self.parabola_rate = torch.tensor([parabola_rate], dtype=torch.float32).repeat(batch_size, 1)
        self.rotation_angle = torch.tensor([rotation_angle], dtype=torch.float32).repeat(batch_size, 1)
        self.patch_size_width = patch_size[1]
        self.patch_size_height = patch_size[0]

    def forward(self, adv_patch):
        width_center = int(self.patch_size_width / 2)
        adv_patch = adv_patch.permute(0, 2, 3, 1)
        tmp = torch.cumsum(adv_patch[:, :, width_center:], dim=2)
        tmp = torch.nn.functional.pad(tmp, (0, 0, 1, 0))
        right_cumsum = torch.transpose(tmp, 1, 2)

        tmp = torch.flip(adv_patch[:, :, :width_center], (-1,))
        tmp = torch.cumsum(tmp, dim=2)
        tmp = torch.nn.functional.pad(tmp, (0, 0, 1, 0))
        left_cumsum = torch.transpose(tmp, 1, 2)

        tmp = np.arange(width_center, self.patch_size_width + 1, dtype=np.float32)
        tmp = torch.from_numpy(tmp)
        tmp = tmp.unsqueeze(0)
        tmp = tf_pre_parabol(tmp, self.parabola_rate, width_center)
        tmp = torch.clamp(tmp - width_center, min=0, max=width_center)
        tmp = torch.round(tmp)
        tmp = tmp.type(torch.int32)
        anchors = tmp.unsqueeze(-1)

        tmp = self.parabola_rate.shape[0]
        tmp = torch.range(start=0, end=tmp - 1)
        tmp = tmp.unsqueeze(-1).unsqueeze(-1)
        anch_inds = torch.tile(tmp, [1, width_center + 1, 1])
        new_anchors = torch.cat([anch_inds, anchors], dim=2)

        tmp = torch.clamp(anchors[:, 1:] - anchors[:, :-1], min=1, max=self.patch_size_width)
        tmp = tmp.type(torch.float32)
        anchors_div = tmp.unsqueeze(-1)

        right_anchors_cumsum = th_gather_nd(right_cumsum, new_anchors)

        return anchors


class PatchApplier(nn.Module):
    def __init__(self, indices):
        super(PatchApplier, self).__init__()
        self.indices = indices

    # def forward(self, img, points):
    #     for detection in points:
    #         plt.imshow(img)
    #         detection[36, 1] = detection[29, 1]
    #         detection[45, 1] = detection[29, 1]
    #         fig = plt.fill(detection[self.indices, 0], detection[self.indices, 1], color=(0.4, 1, 1), edgecolor='white')
    #         plt.scatter(detection[self.indices, 0], detection[self.indices, 1], 2)
    #         plt.show()
    #     print('x')

    def forward(self, img_batch, adv_patch):
        advs = torch.unbind(adv_patch.unsqueeze(1), 1)
        for i, adv in enumerate(advs):
            transforms.ToPILImage()(img_batch[i]).show()
            img_batch = torch.where((adv == 0), img_batch, adv)
            transforms.ToPILImage()(img_batch[i]).show()
        return img_batch


def th_gather_nd(x, coords):
    x = x.contiguous()
    inds = coords.mv(torch.tensor(x.stride(), dtype=torch.long))
    x_gather = torch.index_select(x.contiguous().view(-1), 0, inds)
    return x_gather


def tf_integral(x, a):
    return 0.5 * (x * torch.sqrt(x ** 2 + a) + a * torch.log(torch.abs(x + torch.sqrt(x ** 2 + a))))


def tf_pre_parabol(x, par, width_center):
    x = x - width_center
    prev = 2. * par * (tf_integral(torch.abs(x), 0.25 / (par ** 2)) - tf_integral(0, 0.25 / (par ** 2)))
    return prev + width_center


class ap(nn.Module):
    def __init__(self, patch_size, batch_size, img_size):
        super(ap, self).__init__()
        self.patch_size_width = patch_size[1]
        self.patch_size_height = patch_size[0]
        self.img_size_width = img_size[1]
        self.img_size_height = img_size[0]
        self.batch_size = batch_size
        self.src_lms = self.get_patch_landmarks()

    def forward(self, adv_patch, landmarks):
        patch = adv_patch.repeat((self.batch_size, 1, 1, 1))
        dst_lms = torch.from_numpy(np.concatenate([landmarks[:, 1:16], landmarks[:, 29:30]], axis=1))
        hm_mat = find_homography_dlt(self.src_lms, dst_lms)
        transformed_mask = kornia.warp_perspective(patch, hm_mat, dsize=(self.img_size_height, self.img_size_width))
        # transforms.ToPILImage()(transformed_mask[0]).show()
        return transformed_mask

    def get_patch_landmarks(self):
        lms = np.empty((16, 2))
        lms[0] = [0, 0]
        lms[1] = [0, (self.patch_size_height / 3) * 1]
        lms[2] = [0, (self.patch_size_height / 3) * 2]
        lms[3] = [0, (self.patch_size_height / 3) * 3]
        lms[4] = [(self.patch_size_width / 8) * 1, self.patch_size_height]
        lms[5] = [(self.patch_size_width / 8) * 2, self.patch_size_height]
        lms[6] = [(self.patch_size_width / 8) * 3, self.patch_size_height]
        lms[7] = [(self.patch_size_width / 8) * 4, self.patch_size_height]
        lms[8] = [(self.patch_size_width / 8) * 5, self.patch_size_height]
        lms[9] = [(self.patch_size_width / 8) * 6, self.patch_size_height]
        lms[10] = [(self.patch_size_width / 8) * 7, self.patch_size_height]
        lms[11] = [(self.patch_size_width / 8) * 8, self.patch_size_height]
        lms[12] = [self.patch_size_width, (self.patch_size_height / 3) * 2]
        lms[13] = [self.patch_size_width, (self.patch_size_height / 3) * 1]
        lms[14] = [self.patch_size_width, 0]
        lms[15] = [self.patch_size_width / 2, 0]
        t = torch.tensor(lms.tolist(), dtype=torch.float32).unsqueeze(0)
        t = t.repeat((self.batch_size, 1, 1))
        return t


def read_landmark_106_array(face_lms):
    map = [[1, 2], [3, 4], [5, 6], 7, 9, 11, [12, 13], 14, 16, 18, [19, 20], 21, 23, 25, [26, 27], [28, 29], [30, 31],
           33, 34, 35, 36, 37, 42, 43, 44, 45, 46, 51, 52, 53, 54, 58, 59, 60, 61, 62, 66, 67, 69, 70, 71, 73, 75, 76,
           78, 79, 80, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103]
    pts1 = np.array(face_lms, dtype=np.float)
    pts1 = pts1.reshape((106, 2))
    pts = np.zeros((68, 2))  # map 106 to 68
    for ii in range(len(map)):
        if isinstance(map[ii], list):
            pts[ii] = np.mean(pts1[map[ii]], axis=0)
        else:
            pts[ii] = pts1[map[ii]]
    return pts


class TotalVariation(nn.Module):
    def __init__(self, device) -> None:
        super(TotalVariation, self).__init__()
        self.device = device
        self.uv_mask_src = transforms.ToTensor()(Image.open('../prnet/new_uv.png').convert('L')).to(device).squeeze()
        self.number_of_pixels = torch.count_nonzero(self.uv_mask_src)
        self.save_grads = torch.zeros_like(self.uv_mask_src)

    def forward(self, adv_patch):
        # transforms.ToPILImage()((adv_patch * self.uv_mask_src)[0]).show()
        tv_patch = adv_patch * self.uv_mask_src
        # transforms.ToPILImage()((tv_patch * self.uv_mask_src)[0]).show()
        # tv_patch.register_hook(self.zero_grads)
        loss = total_variation(tv_patch) / self.number_of_pixels
        return loss

    def zero_grads(self, grads):
        self.save_grads = torch.where(grads.sum(dim=1).sum(dim=0).unsqueeze(0).unsqueeze(0) != 0, torch.ones(1, device=self.device), self.save_grads)
        # print(self.save_grads.sum())
