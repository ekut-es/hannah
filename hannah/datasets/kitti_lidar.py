from hannah.datasets.lidar import LidarDataset
from pcdet.utils import object3d_kitti, calibration_kitti
import numpy as np
from skimage import io


class KittiLidarDataset(LidarDataset):

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, augmentor=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger, augmentor=augmentor)

        self.calib = None
        self.root_path = root_path
        self.class_names = class_names
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.augmentor = augmentor

        self.infos = []
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.include_info_data(self.mode)

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne' / f'{idx:0>6}.bin'
        assert lidar_file.exists()
        points = np.fromfile(str(lidar_file), dtype=np.float32)
        return points.reshape(-1, 4)

    def get_image(self, idx):
        img_file = self.root_split_path / 'image_2' / f'{idx:0>6}.png'
        assert img_file.exists()
        image = io.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'image_2' / f'{idx:0>6}.png'
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / f'{idx:0>6}.txt'
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / f'{idx:0>6}.txt'
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)
