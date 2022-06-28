import torch


def render_cy_pt(vertices, new_colors, triangles, b, h, w, device):
    new_image, face_mask = render_texture_pt(vertices, new_colors, triangles, device, b, h, w)
    return face_mask, new_image


def get_mask_from_bb(h, w, device, box):
    points = torch.cartesian_prod(torch.arange(0, h, device=device),
                                  torch.arange(0, w, device=device))
    c1 = (points[:, 0] >= box[2])
    c2 = (points[:, 0] <= box[3])
    c3 = (points[:, 1] >= box[0])
    c4 = (points[:, 1] <= box[1])
    mask = (c1 & c2 & c3 & c4).view(h, w)
    return mask


def get_unique_first_indices(inverse, unique_size):
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    perm = inverse.new_empty(unique_size).scatter_(0, inverse, perm)
    return perm


def get_image_by_vectorization_with_unique_small(bboxes, new_tri_depth, new_tri_tex, new_triangles, vertices, h, w, device):
    depth_sorted, indices = torch.sort(new_tri_depth, descending=True)
    bb_sorted = torch.index_select(input=bboxes, dim=0, index=indices)
    texture_sorted = torch.index_select(input=new_tri_tex, dim=1, index=indices)
    bboxes_unique, inverse = torch.unique(bb_sorted, dim=0, return_inverse=True)
    uni_idx = get_unique_first_indices(inverse, bboxes_unique.size(0))
    depth_sorted = torch.index_select(input=depth_sorted, dim=0, index=uni_idx)
    texture_sorted = torch.index_select(input=texture_sorted, dim=1, index=uni_idx)

    points = torch.cartesian_prod(torch.arange(0, h, device=device),
                                  torch.arange(0, w, device=device))

    points = points.unsqueeze(1).repeat(1, bboxes_unique.shape[0], 1)
    c1 = (points[:, :, 0] >= bboxes_unique[:, 2])
    c2 = (points[:, :, 0] <= bboxes_unique[:, 3])
    c3 = (points[:, :, 1] >= bboxes_unique[:, 0])
    c4 = (points[:, :, 1] <= bboxes_unique[:, 1])

    mask = (c1 & c2 & c3 & c4).view(h, w, -1)

    deep_depth_buffer = torch.zeros([h, w, mask.shape[-1]], dtype=torch.int32, device=device) - 999999.
    dp = torch.where(mask, depth_sorted, deep_depth_buffer).argmax(dim=-1)

    color_img = torch.zeros((3, h, w), device=device)
    color_img = torch.where((mask.sum(dim=-1) == 0), color_img, texture_sorted.T[dp].permute(2, 0, 1))

    mask_img = torch.zeros((1, h, w), device=device)
    mask_img = torch.where((mask.sum(dim=-1) == 0), mask_img, torch.ones(1, device=device))
    return color_img, mask_img


def render_texture_pt(vertices, colors, triangles, device, b, h, w):
    tri_depth = (vertices[:, 2, triangles[0, :]] + vertices[:, 2, triangles[1, :]] + vertices[:, 2, triangles[2, :]]) / 3.
    tri_tex = (colors[:, :, triangles[0, :]] + colors[:, :, triangles[1, :]] + colors[:, :, triangles[2, :]]) / 3.

    umins = torch.max(torch.ceil(torch.min(vertices[:, 0, triangles], dim=1)[0]).type(torch.int), torch.tensor(0, dtype=torch.int))
    umaxs = torch.min(torch.floor(torch.max(vertices[:, 0, triangles], dim=1)[0]).type(torch.int), torch.tensor(w-1, dtype=torch.int))
    vmins = torch.max(torch.ceil(torch.min(vertices[:, 1, triangles], dim=1)[0]).type(torch.int), torch.tensor(0, dtype=torch.int))
    vmaxs = torch.min(torch.floor(torch.max(vertices[:, 1, triangles], dim=1)[0]).type(torch.int), torch.tensor(h-1, dtype=torch.int))

    masks = (umins <= umaxs) & (vmins <= vmaxs)

    image = torch.zeros((b, 3, h, w), device=device)
    face_mask = torch.zeros((b, 1, h, w), device=device)
    for i in range(b):
        bboxes = torch.masked_select(torch.stack([umins[i], umaxs[i], vmins[i], vmaxs[i]]), masks[i]).view(4, -1).T
        new_tri_depth = torch.masked_select(tri_depth[i], masks[i])
        new_tri_tex = torch.masked_select(tri_tex[i], masks[i]).view(3, -1)
        new_triangles = torch.masked_select(triangles, masks[i]).view(3, -1)
        image[i], face_mask[i] = get_image_by_vectorization_with_unique_small(bboxes, new_tri_depth, new_tri_tex, new_triangles, vertices[i], h, w, device)

    return image, face_mask
