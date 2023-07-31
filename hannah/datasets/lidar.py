import numpy as np
import copy
import pickle
import tqdm

try:
    from pcdet.datasets import DatasetTemplate
    from pcdet.datasets.kitti import kitti_utils
    from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
    from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti
    from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
except ImportError as ie:
    print('pcdet import failed, check if pcdet and its dependencies are installed\nfull error:', ie)

#from pointcloud_viewer import PCViewer


class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
        5: 'Truck'
    }
name_to_class = {v: n for n, v in class_to_name.items()}


class LidarDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, augmentor=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.class_names = class_names
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.augmentor = augmentor
        self.infos = []

    def include_info_data(self, mode):
        data_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                data_infos.extend(infos)

        self.infos.extend(data_infos)

    def __len__(self):
        return len(self.sample_id_list)


    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_info(self, sample_idx, has_label=True, count_inside_pts=True):
        info = {}
        pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
        info['point_cloud'] = pc_info

        image_info = {'image_idx': sample_idx,
                      'image_shape': self.get_image_shape(sample_idx)}
        info['image'] = image_info
        calib = self.get_calib(sample_idx)

        P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
        R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
        R0_4x4[3, 3] = 1.
        R0_4x4[:3, :3] = calib.R0
        V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
        calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

        info['calib'] = calib_info

        if has_label:
            obj_list = self.get_label(sample_idx)
            if len(obj_list) == 0:
                annotations = {}
                annotations['name'] = np.array([])
                annotations['truncated'] = np.array([])
                annotations['occluded'] = np.array([])
                annotations['alpha'] = np.array([])
                annotations['bbox'] = np.array([])
                annotations['dimensions'] = np.array([])
                annotations['location'] = np.array([])
                annotations['rotation_y'] = np.array([])
                annotations['score'] = np.array([])
                annotations['difficulty'] = np.array([])
                annotations['index'] = np.array([])
                annotations['gt_boxes_lidar'] = np.array([])
                info['annos'] = annotations
                return info

            annotations = {}
            annotations['name'] = np.array([obj.cls_type for obj in obj_list])
            annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
            annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
            annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
            annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
            annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
            annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
            annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
            annotations['score'] = np.array([obj.score for obj in obj_list])
            annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

            num_objects = len([obj.cls_type for obj in obj_list])
            num_gt = len(annotations['name'])
            index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
            annotations['index'] = np.array(index, dtype=np.int32)

            loc = annotations['location'][:num_objects]
            dims = annotations['dimensions'][:num_objects]
            rots = annotations['rotation_y'][:num_objects]
            loc_lidar = calib.rect_to_lidar(loc)
            l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
            loc_lidar[:, 2] += h[:, 0] / 2
            gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
            annotations['gt_boxes_lidar'] = gt_boxes_lidar

            info['annos'] = annotations

            if count_inside_pts:
                points = self.get_lidar(sample_idx)
                calib = self.get_calib(sample_idx)
                pts_rect = calib.lidar_to_rect(points[:, 0:3])

                fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                pts_fov = points[fov_flag]
                corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                for k in range(num_objects):
                    flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                    num_points_in_gt[k] = flag.sum()
                annotations['num_points_in_gt'] = num_points_in_gt

        return info

    def get_infos(self, has_label=True, count_inside_pts=True):
        infos = []
        for i in tqdm.tqdm(range(len(self.sample_id_list)), 'collecting infos:'):
            infos.append(self.get_info(i, has_label, count_inside_pts))
        return infos

    def evaluation(self, det_annos, class_names, **kwargs):
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]

        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)
        return ap_result_str, ap_dict

    def create_groundtruth_database(self, root_path=None, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(root_path) / ('dense_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    def __getitem__(self, index):
        info = copy.deepcopy(self.infos[index])
        #sample_idx = info['point_cloud']['lidar_idx']
        sample_idx = index
        img_shape = info['image']['image_shape']
        calib = self.get_calib(index)

        input_dict = {
            'frame': self.sample_id_list[sample_idx],
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        points = self.get_lidar(sample_idx)
        assert np.shape(points)[0] > 0
        if self.dataset_cfg.FOV_POINTS_ONLY:
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            points = points[fov_flag]
        input_dict['points'] = points

        # # data augmentation
        # if self.augmentor is not None:
        #     input_dict = self.augmentor.forward(input_dict)

        # point decoration preparation
        if self.dataset_cfg.get('USE_COOP_POINT_DECORATION', False):
            # pad points with zeros for correct input size
            # actual point decoration is done in derived class
            input_dict['points'] = np.hstack((input_dict['points'], np.zeros([np.shape(points)[0], 1])))

        data_dict = self.prepare_data(data_dict=input_dict)

        data_dict['image_shape'] = img_shape

        return data_dict

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=None
        )
        self.split = split
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.id_to_index = {}
        for num, sample_id in enumerate(self.sample_id_list):
            self.id_to_index[sample_id] = num

    # def point_decoration(self, input_dict):
    #     num_extended_point_features = len(self.class_names)
    #     points = input_dict['points']
    #     num_points = np.shape(points)[0]
    #
    #     points = np.hstack([points, np.zeros([num_points, num_extended_point_features])])
    #     for cur_class in self.class_names:
    #         coop_boxes = input_dict['coop_boxes']  # todo change this
    #         coop_ids = input_dict['coop_ids']
    #         if cur_class in gt_names:
    #             cur_boxes = gt_boxes[gt_names == name_to_class[cur_class]]
    #             point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
    #                 torch.from_numpy(points[:, :3]),
    #                 torch.from_numpy(cur_boxes)
    #             ).numpy()
    #
    #             all_point_indices = np.any(point_indices, axis=0)
    #             points[all_point_indices, 4 + name_to_class[cur_class]] = 1
    #
    #
    #     input_dict['points'] = points
    #     return input_dict

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:
        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

        return annos


def create_pkl_file(dataset_cfg, class_names, data_path, save_path, testing=False):
    if testing:
        print('---------------Start to collect validation infos---------------')
        dataset = LidarDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False, testing=testing)
        filename = save_path / 'dense_infos_val.pkl'
        infos = dataset.get_infos(has_label=True, count_inside_pts=True)
        print('val samples:', len(infos))
        with open(filename, 'wb') as f:
            pickle.dump(infos, f)
        print('Dense info val file is saved to %s' % filename)
        return

    print('---------------Start to collect infos---------------')
    dataset = LidarDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('dense_infos_%s.pkl' % train_split)
    val_filename = save_path / ('dense_infos_%s.pkl' % val_split)

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    dense_infos_train = dataset.get_infos(has_label=True, count_inside_pts=True)
    print('train samples:', len(dense_infos_train))
    with open(train_filename, 'wb') as f:
        pickle.dump(dense_infos_train, f)
    print('Dense info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    dense_infos_val = dataset.get_infos(has_label=True, count_inside_pts=True)
    print('val samples:', len(dense_infos_val))
    with open(val_filename, 'wb') as f:
        pickle.dump(dense_infos_val, f)
    print('Dense info val file is saved to %s' % val_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(save_path, train_filename, split=train_split)
    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import yaml
    from pathlib import Path
    from easydict import EasyDict
    dataset_path = Path('/media/sven/7cbad348-a2fb-46e0-9461-651039557046/MA/hannah/datasets/dense_rain/')

    dataset_config = EasyDict(yaml.safe_load(open('/home/sven/MA/hannah/hannah/conf/dataset/dense_lidar_voxel.yaml')))
    create_pkl_file(
            dataset_cfg=dataset_config,
            class_names=['PassengerCar', 'Pedestrian', 'RidableVehicle'],
            data_path=dataset_path,
            save_path=dataset_path,
            testing=True)

