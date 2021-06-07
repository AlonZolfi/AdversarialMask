import torch


def render_cy_pt(vertices, new_colors, triangles, b, h, w, device):
    vis_colors = torch.ones((b, 1, vertices.shape[-1]), device=device)
    # vis_colors = torch.ones((1, vertices.shape[-1]), device=device)
    new_image = render_texture_pt(vertices, new_colors, triangles, device, b, h, w, 3)
    # new_image = render_texture_pt(vertices.squeeze(0), new_colors.squeeze(0), triangles, device, b, h, w, 3)
    face_mask = render_texture_pt(vertices, vis_colors, triangles, device, b, h, w, 1)
    # face_mask = render_texture_pt(vertices.squeeze(0), vis_colors, triangles, device, b, h, w, 1)
    return face_mask, new_image


def render_texture_pt(vertices, colors, triangles, device, b, h, w, c = 3):
    ''' render mesh by z buffer
    Args:
        vertices: 3 x nver
        colors: 3 x nver
        triangles: 3 x ntri
        h: height
        w: width
    '''
    # initial
    # image = torch.zeros((h, w, c))
    # image1 = torch.zeros((h, w, c))
    # image2 = torch.zeros((h, w, c), device=device)
    # image2 = torch.zeros((b, h, w, c), device=device)

    # depth_buffer = torch.zeros([h, w]) - 999999.
    # depth_buffer1 = torch.zeros([h, w], device=device) - 999999.
    # depth_buffer1 = torch.zeros([b, h, w], device=device) - 999999.
    # depth_buffer2 = torch.zeros([h, w], device=device) - 999999.
    # triangle depth: approximate the depth to the average value of z in each vertex(v0, v1, v2), since the vertices are closed to each other
    tri_depth = (vertices[:, 2, triangles[0, :]] + vertices[:, 2, triangles[1, :]] + vertices[:, 2, triangles[2, :]]) / 3.
    tri_tex = (colors[:, :, triangles[0, :]] + colors[:, :, triangles[1, :]] + colors[:, :, triangles[2, :]]) / 3.

    umins = torch.max(torch.ceil(torch.min(vertices[:, 0, triangles], dim=1)[0]).type(torch.int), torch.tensor(0, dtype=torch.int))
    umaxs = torch.min(torch.floor(torch.max(vertices[:, 0, triangles], dim=1)[0]).type(torch.int), torch.tensor(w-1, dtype=torch.int))
    vmins = torch.max(torch.ceil(torch.min(vertices[:, 1, triangles], dim=1)[0]).type(torch.int), torch.tensor(0, dtype=torch.int))
    vmaxs = torch.min(torch.floor(torch.max(vertices[:, 1, triangles], dim=1)[0]).type(torch.int), torch.tensor(h-1, dtype=torch.int))

    masks = (umins <= umaxs) & (vmins <= vmaxs)
    # bboxes = torch.masked_select(torch.stack([umin, umax, vmin, vmax]), mask).view(b, 4, -1).permute(0, 2, 1)
    # points = torch.cartesian_prod(torch.arange(0, h, device=device), torch.arange(0, w, device=device)).unsqueeze(0).repeat(bboxes.shape[0], 1, 1)
    # c1 = (points[:, 0] >= bboxes[:, 0]).type(torch.int)
    # c2 = (points[:, 0] <= bboxes[:, 1]).type(torch.int)
    # c3 = (points[:, 1] >= bboxes[:, 2]).type(torch.int)
    # c4 = (points[:, 1] <= bboxes[:, 3]).type(torch.int)

    # indices = torch.masked_select(torch.stack([umin, umax, vmin, vmax]), mask).view(4, -1)
    # points = torch.cartesian_prod(torch.arange(0, h, device=device), torch.arange(0, w, device=device))
    # old_points = points
    # points = points.unsqueeze(1)
    # points = points.repeat(1, bboxes.shape[0], 1)
    # c1 = (points[:, :, 0] >= bboxes[:, 0]).type(torch.int)
    # c2 = (points[:, :, 0] <= bboxes[:, 1]).type(torch.int)
    # c3 = (points[:, :, 1] >= bboxes[:, 2]).type(torch.int)
    # c4 = (points[:, :, 1] <= bboxes[:, 3]).type(torch.int)
    #
    # mask = c1 + c2 + c3 + c4
    # mask = torch.nonzero((mask == 4).sum(dim=-1)).squeeze()

    # new_triangles = torch.masked_select(triangles, mask).view(b, 3, -1)
    # new_tri_depth = torch.masked_select(tri_depth, mask).view(b, -1)
    # new_tri_tex = torch.masked_select(tri_tex, mask).view(b, c, -1)

    image = torch.zeros((b, c, h, w), device=device)
    for i in range(b):
        bboxes = torch.masked_select(torch.stack([umins[i], umaxs[i], vmins[i], vmaxs[i]]), masks[i]).view(4, -1).T
        print(bboxes.shape[0])
        points = torch.cartesian_prod(torch.arange(0, h, device=device),
                                      torch.arange(0, w, device=device))
        points = points.unsqueeze(1).repeat(1, bboxes.shape[0], 1)
        c1 = (points[:, :, 0] >= bboxes[:, 2])
        c2 = (points[:, :, 0] <= bboxes[:, 3])
        c3 = (points[:, :, 1] >= bboxes[:, 0])
        c4 = (points[:, :, 1] <= bboxes[:, 1])
        del points, bboxes
        torch.cuda.empty_cache()
        mask = (c1 & c2 & c3 & c4).view(h, w, -1)

        new_tri_depth = torch.masked_select(tri_depth[i], masks[i])
        deep_depth_buffer = torch.zeros([h, w, mask.shape[-1]], dtype=torch.int32, device=device) - 999999.
        dp = torch.where(mask, new_tri_depth, deep_depth_buffer).argmax(dim=-1)
        del new_tri_depth, deep_depth_buffer
        new_tri_tex = torch.masked_select(tri_tex[i], masks[i]).view(c, -1)
        image[i] = torch.where((mask.sum(dim=-1) == 0), image[i], new_tri_tex.T[dp].permute(2, 0, 1))
        # new_triangles = torch.masked_select(triangles, masks[i]).view(3, -1)

        '''for j in range(bboxes.shape[0]):
            umin = bboxes[j, 0].item()
            umax = bboxes[j, 1].item()
            vmin = bboxes[j, 2].item()
            vmax = bboxes[j, 3].item()
            # uv_vector = torch.cartesian_prod(torch.arange(umin, umax+1, device=device),
            #                                  torch.arange(vmin, vmax+1, device=device))
            condition = (new_tri_depth[j] > depth_buffer1[i, vmin:vmax+1, umin:umax+1])
            # condition = (new_tri_depth[j] > depth_buffer1[i, vmin:vmax+1, umin:umax+1]) & \
            #             (arePointsInTri_pt(uv_vector, vertices[i, :2, new_triangles[:, j]].unsqueeze(0), umax-umin+1, vmax-vmin+1))
            depth_buffer1[i, vmin:vmax+1, umin:umax+1] = torch.where(condition,
                                                                     new_tri_depth[j].repeat(condition.shape),
                                                                     depth_buffer1[i, vmin:vmax+1, umin:umax+1])

            image2[i, vmin:vmax + 1, umin:umax + 1, :] = torch.where(condition.unsqueeze(-1).repeat(1, 1, c),
                                                                     new_tri_tex[:, j].repeat(condition.shape).view(condition.shape[0], condition.shape[1], -1),
                                                                     image2[i, vmin:vmax + 1, umin:umax + 1, :])
        print((depth_buffer == depth_buffer1[i]).all())
        print((image[i].permute(1,2,0) == image2[i]).all())'''
    return image


def arePointsInTri_pt(points, tri_points, u_range, v_range):
    ''' Judge whether the point is in the triangle
    Method:
        http://blackpawn.com/texts/pointinpoly/
    Args:
        point: [u, v] or [x, y]
        tri_points: three vertices(2d points) of a triangle. 2 coords x 3 vertices
    Returns:
        bool: true for in triangle
    '''
    tp = tri_points

    # vectors
    v0 = tp[:, :, 2] - tp[:, :, 0]
    v1 = tp[:, :, 1] - tp[:, :, 0]
    v2 = points - tp[:, :, 0]

    # dot products
    dot00 = torch.matmul(v0, v0.T)
    dot01 = torch.matmul(v0, v1.T)
    dot02 = torch.matmul(v0, v2.T)
    dot11 = torch.matmul(v1, v1.T)
    dot12 = torch.matmul(v1, v2.T)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    # check if point in triangle
    cond = (u >= 0) & (v >= 0) & (u + v < 1)
    cond = cond.squeeze(0).view(u_range, v_range).T
    return cond


def isPointInTri_pt(point, tri_points):
    ''' Judge whether the point is in the triangle
    Method:
        http://blackpawn.com/texts/pointinpoly/
    Args:
        point: [u, v] or [x, y]
        tri_points: three vertices(2d points) of a triangle. 2 coords x 3 vertices
    Returns:
        bool: true for in triangle
    '''
    tp = tri_points

    # vectors
    v0 = tp[:, 2] - tp[:, 0]
    v1 = tp[:, 1] - tp[:, 0]
    v2 = point - tp[:, 0]

    # dot products
    dot00 = torch.matmul(v0.T, v0)
    dot01 = torch.matmul(v0.T, v1)
    dot02 = torch.matmul(v0.T, v2)
    dot11 = torch.matmul(v1.T, v1)
    dot12 = torch.matmul(v1.T, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    # check if point in triangle
    return (u >= 0) & (v >= 0) & (u + v < 1)