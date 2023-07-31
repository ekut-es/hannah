from hannah.datasets.lidar import LidarDataset
from pcdet.utils import object3d_kitti, calibration_kitti
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
import numpy as np
from skimage import io
import pickle
import tqdm
import glob
import copy

DENSE_TO_KITTI_LABEL = {'PassengerCar': 'Car',
                        'Pedestrian': 'Pedestrian',
                        'RidableVehicle': 'Cyclist',
                        'LargeVehicle': 'Van',
                        'Vehicle': 'Van',
                        'DontCare': 'DontCare'}

OBJECT_TYPE_TO_ID = {'Car': 1,
                     'Pedestrian': 2,
                     'Cyclist': 3,
                     'Van': 4}

class DenseLidarDataset(LidarDataset):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, augmentor=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger, augmentor=augmentor)

        self.calib = None
        self.root_path = root_path
        self.class_names = class_names
        self.sensor_return_type = self.dataset_cfg.RETURN_TYPE
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.augmentor = augmentor
        self.infos = []

        self.split_files = self.dataset_cfg.DATA_SPLIT[self.mode]

        self.sample_id_list = []
        for split_file in self.split_files:
            split_path = self.root_path / 'splits' / (split_file + '.txt')
            assert split_path.exists()
            self.sample_id_list += ['_'.join(x.strip().split(',')) for x in open(split_path).readlines()]

        self.id_to_index = {}
        for num, sample_id in enumerate(self.sample_id_list):
            self.id_to_index[sample_id] = num

        self.include_info_data(self.mode)

        # remove frames without relevand objects
        for idx in reversed(range(len(self.infos))):
            if not any(x in self.infos[idx]['annos']['name'] for x in self.class_names):
                del self.infos[idx]
                del self.sample_id_list[idx]

    def set_split(self, split):
        split_path = self.root_path / 'splits' / (split + '.txt')
        assert split_path.exists()
        self.sample_id_list = ['_'.join(x.strip().split(',')) for x in open(split_path).readlines()]
        self.id_to_index = {}
        for num, sample_id in enumerate(self.sample_id_list):
            self.id_to_index[sample_id] = num

    def get_infos(self, has_label=True, count_inside_pts=True):
        infos = []
        for i in tqdm.tqdm(self.sample_id_list):
            infos.append(self.get_info(i, has_label, count_inside_pts))
        return infos

    def get_lidar(self, idx):
        lidar_file = self.root_path / ('lidar_hdl64_%s' % self.sensor_return_type) / ('%s.bin' % idx)
        assert lidar_file.exists()
        points = np.fromfile(str(lidar_file), dtype=np.float32)
        return points.reshape(-1, 5)[:, :4]

    def get_image(self, idx):
        img_file = self.root_path / 'cam_stereo_left_lut' / '000000.png' #('%s.png' % idx)
        assert img_file.exists()
        image = io.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_image_shape(self, idx):
        img_file = self.root_path / 'cam_stereo_left_lut' / '000000.png' #('%s.png' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_path / 'cam_left_labels_TMP' / ('%s.txt' % idx)
        assert label_file.exists()
        labels = object3d_kitti.get_objects_from_label(label_file)
        for label in labels:
            label.cls_type = DENSE_TO_KITTI_LABEL.get(label.cls_type, 'ignore')
            label.cls_id = OBJECT_TYPE_TO_ID.get(label.cls_type, label.cls_type)
        return labels

    def get_calib(self, idx):
        if self.calib is not None:
            return self.calib
        else:
            calib_file = self.root_path / 'calib' / 'calib.txt'
            assert calib_file.exists()
            return calibration_kitti.Calibration(calib_file)

    def evaluation(self, det_annos, class_names, **kwargs):
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)
        print(ap_result_str)
        return ap_result_str, ap_dict


def create_pkl_file(dataset_cfg, class_names, data_path, save_path):
    dataset = DenseLidarDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    print(str(data_path / 'splits' / '*.txt'))
    print(glob.glob(str(data_path / 'splits' / '*.txt')))
    for split in glob.glob(str(data_path / 'splits' / '*.txt')):
        split_name = split.split('/')[-1].split('.')[0]
        print('---- start collecting infos for split: '+split_name+' ----')
        dataset.set_split(split_name)
        filename = save_path / ('infos_%s.pkl' % split_name)
        infos = dataset.get_infos(has_label=True, count_inside_pts=True)
        with open(filename, 'wb') as f:
            pickle.dump(infos, f)
        print('info file is saved to %s' % filename)


if __name__ == '__main__':
    import yaml
    from pathlib import Path
    from easydict import EasyDict
    dataset_path = Path('../../datasets/dense')

    dataset_config = EasyDict(yaml.safe_load(open('../conf/dataset/dense_lidar.yaml')))
    create_pkl_file(
            dataset_cfg=dataset_config,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=dataset_path,
            save_path=dataset_path)
