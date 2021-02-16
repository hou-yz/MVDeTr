import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import re
from torchvision.datasets import VisionDataset

intrinsic_camera_matrix_filenames = ['intr_Camera1.xml', 'intr_Camera2.xml', 'intr_Camera3.xml', 'intr_Camera4.xml',
                                     'intr_Camera5.xml', 'intr_Camera6.xml']
extrinsic_camera_matrix_filenames = ['extr_Camera1.xml', 'extr_Camera2.xml', 'extr_Camera3.xml', 'extr_Camera4.xml',
                                     'extr_Camera5.xml', 'extr_Camera6.xml']


class MultiviewX(VisionDataset):
    def __init__(self, root):
        super().__init__(root)
        # image of shape C,H,W (C,N_row,N_col); xy indexging; x,y (w,h) (n_col,n_row)
        # MultiviewX has xy-indexing: H*W=640*1000, thus x is \in [0,1000), y \in [0,640)
        # MultiviewX has consistent unit: meter (m) for calibration & pos annotation
        self.__name__ = 'MultiviewX'
        self.img_shape, self.worldgrid_shape = [1080, 1920], [640, 1000]  # H,W; N_row,N_col
        self.num_cam, self.num_frame = 6, 400
        # world x,y correspond to w,h
        self.indexing = 'xy'
        self.world_indexing_from_xy_mat = np.eye(3)
        self.world_indexing_from_ij_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        # image is in xy indexing by default
        self.img_xy_from_xy_mat = np.eye(3)
        self.img_xy_from_ij_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        # unit in meters
        self.worldcoord_unit = 1
        self.worldcoord_from_worldgrid_mat = np.array([[0.025, 0, 0], [0, 0.025, 0], [0, 0, 1]])
        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])

    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        grid_x = pos % 1000
        grid_y = pos // 1000
        return np.array([[grid_x], [grid_y]], dtype=int).reshape([2, -1])

    def get_pos_from_worldgrid(self, worldgrid):
        grid_x, grid_y = worldgrid[0, :], worldgrid[1, :]
        return grid_x + grid_y * 1000

    def get_worldgrid_from_worldcoord(self, world_coord):
        # datasets default unit: centimeter & origin: (-300,-900)
        coord_x, coord_y = world_coord[0, :], world_coord[1, :]
        grid_x = coord_x * 40
        grid_y = coord_y * 40
        return np.array([[grid_x], [grid_y]], dtype=int).reshape([2, -1])

    def get_worldcoord_from_worldgrid(self, worldgrid):
        # datasets default unit: centimeter & origin: (-300,-900)
        grid_x, grid_y = worldgrid[0, :], worldgrid[1, :]
        coord_x = grid_x / 40
        coord_y = grid_y / 40
        return np.array([[coord_x], [coord_y]]).reshape([2, -1])

    def get_worldcoord_from_pos(self, pos):
        grid = self.get_worldgrid_from_pos(pos)
        return self.get_worldcoord_from_worldgrid(grid)

    def get_pos_from_worldcoord(self, world_coord):
        grid = self.get_worldgrid_from_worldcoord(world_coord)
        return self.get_pos_from_worldgrid(grid)

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        intrinsic_camera_path = os.path.join(self.root, 'calibrations', 'intrinsic')
        fp_calibration = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                      intrinsic_camera_matrix_filenames[camera_i]),
                                         flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = fp_calibration.getNode('camera_matrix').mat()
        fp_calibration.release()

        extrinsic_camera_path = os.path.join(self.root, 'calibrations', 'extrinsic')
        fp_calibration = cv2.FileStorage(os.path.join(extrinsic_camera_path,
                                                      extrinsic_camera_matrix_filenames[camera_i]),
                                         flags=cv2.FILE_STORAGE_READ)
        rvec, tvec = fp_calibration.getNode('rvec').mat().squeeze(), fp_calibration.getNode('tvec').mat().squeeze()
        fp_calibration.release()

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_matrix = np.array(tvec, dtype=np.float).reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))

        return intrinsic_matrix, extrinsic_matrix

    def read_pom(self):
        bbox_by_pos_cam = {}
        cam_pos_pattern = re.compile(r'(\d+) (\d+)')
        cam_pos_bbox_pattern = re.compile(r'(\d+) (\d+) ([-\d]+) ([-\d]+) (\d+) (\d+)')
        with open(os.path.join(self.root, 'rectangles.pom'), 'r') as fp:
            for line in fp:
                if 'RECTANGLE' in line:
                    cam, pos = map(int, cam_pos_pattern.search(line).groups())
                    if pos not in bbox_by_pos_cam:
                        bbox_by_pos_cam[pos] = {}
                    if 'notvisible' in line:
                        bbox_by_pos_cam[pos][cam] = None
                    else:
                        cam, pos, left, top, right, bottom = map(int, cam_pos_bbox_pattern.search(line).groups())
                        bbox_by_pos_cam[pos][cam] = [max(left, 0), max(top, 0),
                                                     min(right, 1920 - 1), min(bottom, 1080 - 1)]
        return bbox_by_pos_cam


def test():
    from multiview_detector.utils.projection import get_worldcoord_from_imagecoord
    dataset = MultiviewX(os.path.expanduser('~/Data/MultiviewX'), )
    pom = dataset.read_pom()

    for cam in range(dataset.num_cam):
        head_errors, foot_errors = [], []
        for pos in range(0, np.product(dataset.worldgrid_shape), 16):
            bbox = pom[pos][cam]
            foot_wc = dataset.get_worldcoord_from_pos(pos)
            if bbox is None:
                continue
            foot_ic = np.array([[(bbox[0] + bbox[2]) / 2, bbox[3]]]).T
            head_ic = np.array([[(bbox[0] + bbox[2]) / 2, bbox[1]]]).T
            p_foot_wc = get_worldcoord_from_imagecoord(foot_ic, dataset.intrinsic_matrices[cam],
                                                       dataset.extrinsic_matrices[cam])
            p_head_wc = get_worldcoord_from_imagecoord(head_ic, dataset.intrinsic_matrices[cam],
                                                       dataset.extrinsic_matrices[cam], z=1.8 / dataset.worldcoord_unit)
            head_errors.append(np.linalg.norm(p_head_wc - foot_wc))
            foot_errors.append(np.linalg.norm(p_foot_wc - foot_wc))
            pass

        print(f'average head error: {np.average(head_errors) * dataset.worldcoord_unit}, '
              f'average foot error: {np.average(foot_errors) * dataset.worldcoord_unit} (world meters)')
        pass
    pass


if __name__ == '__main__':
    test()
