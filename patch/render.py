import torch


def render_cy_pt(vertices, new_colors, triangles, b, h, w):
    vis_colors = torch.ones((1, vertices.shape[-1]))
    new_image = render_texture_pt(vertices.squeeze(0), new_colors.squeeze(0), triangles, b, h, w, 3)
    face_mask = render_texture_pt(vertices.squeeze(0), vis_colors, triangles, b, h, w, 1)
    return face_mask, new_image


def render_texture_pt(vertices, colors, triangles, b, h, w, c = 3):
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
    image1 = torch.zeros((h, w, c))

    # depth_buffer = torch.zeros([h, w]) - 999999.
    depth_buffer1 = torch.zeros([h, w]) - 999999.
    # triangle depth: approximate the depth to the average value of z in each vertex(v0, v1, v2), since the vertices are closed to each other
    tri_depth = (vertices[2, triangles[0,:]] + vertices[2,triangles[1,:]] + vertices[2, triangles[2,:]])/3.
    tri_tex = (colors[:, triangles[0, :]] + colors[:,triangles[1, :]] + colors[:, triangles[2, :]])/3.

    for i in range(triangles.shape[1]):
        tri = triangles[:, i] # 3 vertex indices

        # the inner bounding box
        umin = max(int(torch.ceil(torch.min(vertices[0, tri]))), 0)
        umax = min(int(torch.floor(torch.max(vertices[0, tri]))), w-1)

        vmin = max(int(torch.ceil(torch.min(vertices[1, tri]))), 0)
        vmax = min(int(torch.floor(torch.max(vertices[1, tri]))), h-1)

        if umax < umin or vmax < vmin:
            continue

        uv_vector = torch.cartesian_prod(torch.arange(umin, umax+1), torch.arange(vmin, vmax+1))
        condition = (tri_depth[i] > depth_buffer1[vmin:vmax+1, umin:umax+1]) & \
                    (arePointsInTri_pt(uv_vector, vertices[:2, tri].unsqueeze(0), umax-umin+1, vmax-vmin+1))
        depth_buffer1[vmin:vmax+1, umin:umax+1] = torch.where(condition, tri_depth[i].repeat(condition.shape), depth_buffer1[vmin:vmax+1, umin:umax+1])
        for j in range(image1.shape[-1]):
            image1[vmin:vmax+1, umin:umax+1, j] = torch.where(condition, tri_tex[j, i].repeat(condition.shape), image1[vmin:vmax+1, umin:umax+1, j])

        # for u in range(umin, umax+1):
        #     for v in range(vmin, vmax+1):
        #         if tri_depth[i] > depth_buffer[v, u] and isPointInTri_pt(torch.tensor([u, v]), vertices[:2, tri]):
        #             depth_buffer[v, u] = tri_depth[i]
        #             image[v, u, :] = tri_tex[:, i]
        # if torch.abs( depth_buffer1[vmin:vmax+1, umin:umax+1] -  depth_buffer[vmin:vmax+1, umin:umax+1]).sum() > 0:
        #     print('depth')
        # if torch.abs( image1[vmin:vmax+1, umin:umax+1] -  image[vmin:vmax+1, umin:umax+1]).sum() > 0:
        #     print('image')
    return image1


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