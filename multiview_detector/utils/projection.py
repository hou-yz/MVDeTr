import numpy as np


def project_2d_points(project_mat, input_points):
    vertical_flag = 0
    if input_points.shape[1] == 2:
        vertical_flag = 1
        input_points = np.transpose(input_points)
    input_points = np.concatenate([input_points, np.ones([1, input_points.shape[1]])], axis=0)
    output_points = project_mat @ input_points
    output_points = output_points[:2, :] / output_points[2, :]
    if vertical_flag:
        output_points = np.transpose(output_points)
    return output_points


def get_worldcoord_from_imagecoord(image_coord, intrinsic_mat, extrinsic_mat, z=0):
    project_mat = get_worldcoord_from_imgcoord_mat(intrinsic_mat, extrinsic_mat, z)
    return project_2d_points(project_mat, image_coord)


def get_imagecoord_from_worldcoord(world_coord, intrinsic_mat, extrinsic_mat, z=0):
    project_mat = get_imgcoord_from_worldcoord_mat(intrinsic_mat, extrinsic_mat, z)
    return project_2d_points(project_mat, world_coord)


def get_imgcoord_from_worldcoord_mat(intrinsic_mat, extrinsic_mat, z=0):
    """image of shape C,H,W (C,N_row,N_col); xy indexging; x,y (w,h) (n_col,n_row)
    world of shape N_row, N_col; indexed as specified in the dataset attribute (xy or ij)
    z in meters by default
    """
    threeD2twoD = np.array([[1, 0, 0], [0, 1, 0], [0, 0, z], [0, 0, 1]])
    project_mat = intrinsic_mat @ extrinsic_mat @ threeD2twoD
    return project_mat


def get_worldcoord_from_imgcoord_mat(intrinsic_mat, extrinsic_mat, z=0):
    """image of shape C,H,W (C,N_row,N_col); xy indexging; x,y (w,h) (n_col,n_row)
    world of shape N_row, N_col; indexed as specified in the dataset attribute (xy or ij)
    z in meters by default
    """
    project_mat = np.linalg.inv(get_imgcoord_from_worldcoord_mat(intrinsic_mat, extrinsic_mat, z))
    return project_mat
