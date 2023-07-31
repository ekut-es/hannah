import glob
import os
from hannah.datasets.lidar import LidarDataset
from pcdet.utils import calibration_kitti, box_utils
from pcdet.datasets.kitti.kitti_utils import transform_annotations_to_kitti_format
from pcdet.datasets.kitti.kitti_object_eval_python.eval import get_official_eval_result
import numpy as np
import copy
import random
import pickle
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
import torch
import transforms3d
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from pointcloud_viewer import PCViewer

def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(',')
        self.cls_type = 'Car' if label[0] == '1' else 'DontCare'
        self.cls_id = cls_type_to_id(self.cls_type)
        self.loc = np.array((float(label[2]), float(label[3]), float(label[4])), dtype=np.float32)
        self.w = float(label[5])
        self.h = float(label[6])
        self.l = float(label[7])
        self.ry = -float(label[8]) + np.pi/2
        self.yaw = float(label[8])
        self.pitch = float(label[9])
        self.roll = float(label[10])


class ScopeLidarDataset(LidarDataset):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, augmentor=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger, augmentor=augmentor)
        self.root_path = root_path
        print(root_path)
        self.class_names = class_names
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.augmentor = augmentor

        self.infos = []
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.sample_id_list = []
        self.frame_order = None
        self.first_frames_to_delete = 30

        if self.dataset_cfg.SAVE_DETECTIONS:
            if training:
                return

            vehicle = self.dataset_cfg.SAVE_DETECTION_VEHICLE
            self.sample_id_list += [vehicle + '/' + os.path.basename(x).split('.')[0] for x in sorted(glob.glob(str(self.root_path / vehicle / "lidar/*bin")))][self.first_frames_to_delete:]
            self.infos.extend(self.get_infos())
        else:
            for vehicle in self.split:
                self.sample_id_list += [vehicle + '/' + os.path.basename(x).split('.')[0] for x in sorted(glob.glob(str(self.root_path / vehicle / "lidar/*bin")))][self.first_frames_to_delete:]

            self.include_info_data(self.mode)
            self.vehicle_states = self.get_vehicle_states()
            self.coop_detections = self.get_coop_detections()
            self.remove_global_ego_coords()
            if training:
                self.frame_order = list(range(len(self.sample_id_list)))
                random.Random(13).shuffle(self.frame_order)
                self.infos = [self.infos[i] for i in self.frame_order]
                self.sample_id_list = [self.sample_id_list[i] for i in self.frame_order]

    def get_vehicle_states(self):
        all_infos = []
        for info_path in self.dataset_cfg.INFO_PATH['all']:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                all_infos.extend(infos)
        def get_ego_anno(anno):
            return {'location': anno['location'][0], 'rotation': np.array([anno['roll'][0], anno['pitch'][0], anno['yaw'][0]])}

        vehicle_states = [get_ego_anno(x['annos']) for x in all_infos]
        split_size = int(len(vehicle_states) / len(self.dataset_cfg.DATA_SPLIT['all']))
        vehicle_states = [vehicle_states[split_size * i:split_size * (i + 1):] for i in range(len(self.dataset_cfg.DATA_SPLIT['all']))]
        return vehicle_states

    def remove_global_ego_coords(self):
        for info in self.infos:
            for key, value in info['annos'].items():
                info['annos'][key] = value[1:]




    def get_lidar(self, idx):
        vehicle = self.sample_id_list[idx].split('/')[0]
        lidar_file = self.root_path / vehicle / 'lidar' / os.path.basename((self.sample_id_list[idx]+'.bin'))
        assert lidar_file.exists()
        points = np.fromfile(str(lidar_file), dtype=np.float32)
        return points.reshape(-1, 4)

    def get_coop_detections(self):
        coop_detections = []
        for vehicle in self.dataset_cfg.DATA_SPLIT['all']:
            detection_path = self.root_path / (vehicle+'_DETECTIONS.pkl')
            if not detection_path.exists():
                continue
            with open(detection_path, 'rb') as f:
                detections = pickle.load(f)
                coop_detections.append(detections['detections'])
        return coop_detections

    def get_image(self, idx):
        return None

    def get_image_shape(self, idx):
        return np.array([0, 0], dtype=np.int32)

    def get_label(self, idx):
        sample = os.path.basename(self.sample_id_list[idx]+'.csv')
        vehicle = self.sample_id_list[idx].split('/')[0]
        label_file = self.root_path / vehicle / 'road_user_gt' / sample
        assert label_file.exists()
        global_gt_objects = self.get_objects_from_label(label_file)

        return global_gt_objects

    def get_calib(self, idx):
        calib = {'P2': np.eye(3, 4),
                 'R0': np.eye(3),
                 'Tr_velo2cam': np.eye(3, 4)}

        calib['Tr_velo2cam'][2, 3] = 1.8  # sensor offset
        return calibration_kitti.Calibration(calib)

    def evaluation(self, det_annos, class_names, **kwargs):
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]
        for gt_anno, det_anno in zip(eval_gt_annos, eval_det_annos):
            # mask ground truth that is out of range
            gt_mask = box_utils.mask_boxes_outside_range_numpy(
                gt_anno['gt_boxes_lidar'], self.point_cloud_range,
                min_num_corners=1,
                use_center_to_filter=True
            )
            gt_mask = np.all(np.stack((gt_mask, gt_anno['name'] == 'Car'), axis=0), axis=0) # mask all none Car labels

            for key, value in gt_anno.items():
                gt_anno[key] = value[gt_mask]

        if self.dataset_cfg.get('LATE_FUSION', False):
            for detection in eval_det_annos:
                ego_det = np.concatenate((detection['boxes_lidar'], detection['score'].reshape(-1, 1)), axis=1)
                coop_det = np.concatenate((detection['coop_boxes'], detection['coop_scores'].reshape(-1, 1)), axis=1)
                matches = self.detection_matching(ego_det, coop_det, detection['coop_vehicle_ids'])
                fused_detections = self.weighted_mean_fusion(matches)
                # mask out of range detections
                det_mask = box_utils.mask_boxes_outside_range_numpy(
                    fused_detections[:, :7], self.point_cloud_range,
                    min_num_corners=1,
                    use_center_to_filter=True
                )
                detection['boxes_lidar'] = fused_detections[det_mask, :7]
                detection['score'] = fused_detections[det_mask, 7]
                detection['name'] = np.array(['Car']*np.shape(detection['score'])[0])

        transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti={'Car': 'Car'})
        transform_annotations_to_kitti_format(eval_gt_annos, map_name_to_kitti={'Car': 'Car'})

        ap_result_str, ap_dict = get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)
        return ap_result_str, ap_dict

    def detection_matching(self, ego_detections, coop_detections, coop_vehicle_ids):
        MAX_MATCH_DIST = 2  # set the max dist of matched detection to 2m
        ego_detected_pos = ego_detections[:, :3]
        coop_detected_pos = coop_detections[:, :3]
        matches = [[x] for x in ego_detections]
        for vehicle_idx in np.unique(coop_vehicle_ids):
            cur_coop_pos = coop_detected_pos[vehicle_idx == coop_vehicle_ids, :]
            cost = cdist(ego_detected_pos, cur_coop_pos, metric='euclidean')
            cur_matches = linear_sum_assignment(cost)
            cur_coop_detections = coop_detections[vehicle_idx == coop_vehicle_ids]

            # new track for left over coop det
            if np.shape(cur_coop_pos)[0] > np.shape(ego_detected_pos)[0]:
                unmached = np.delete(cur_coop_detections, cur_matches[1], axis=0)
                for um in unmached:
                    matches.append([um])
                ego_detected_pos = np.concatenate((ego_detected_pos, unmached[:, :3]), axis=0)

            for ego_det_idx, coop_det_idx in zip(cur_matches[0], cur_matches[1]):
                dist = cost[ego_det_idx, coop_det_idx]
                coop_det = cur_coop_detections[coop_det_idx]

                # match to existing tracks
                if dist <= MAX_MATCH_DIST:
                    matches[ego_det_idx].append(coop_det)

                # open new track for false matched
                else:
                    matches.append([coop_det])
                    ego_detected_pos = np.vstack((ego_detected_pos, coop_det[:3]))

        return matches

    def weighted_mean_fusion(self, matches):
        fused_boxes = []
        for m in matches:
            num_boxes = len(m)
            if num_boxes == 1:
                fused_boxes.append(m[0])
            else:
                m = np.vstack(m)
                m[:, :7] *= m[:, 7, np.newaxis]
                m = np.sum(m, axis=0)
                m[:7] /= m[7]
                m[7] /= num_boxes
                fused_boxes.append(m)
        return np.vstack(fused_boxes)



    def get_info(self, sample_idx, has_label=True, count_inside_pts=True):
        info = {}
        pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
        info['point_cloud'] = pc_info

        image_info = {'image_idx': sample_idx,
                      'image_shape': self.get_image_shape(sample_idx)}
        info['image'] = image_info
        calib = self.get_calib(sample_idx)

        calib_info = {'P2': np.eye(4), 'R0_rect': np.eye(4), 'Tr_velo_to_cam': np.eye(4)}

        info['calib'] = calib_info

        if has_label:
            obj_list = self.get_label(sample_idx)
            if len(obj_list) == 0:
                annotations = {}
                annotations['name'] = np.array([])
                annotations['dimensions'] = np.array([])
                annotations['location'] = np.array([])
                annotations['rotation_y'] = np.array([])
                annotations['yaw'] = np.array([])
                annotations['pitch'] = np.array([])
                annotations['roll'] = np.array([])
                annotations['index'] = np.array([])
                annotations['gt_boxes_lidar'] = np.array([])
                info['annos'] = annotations
                return info

            annotations = {}
            annotations['name'] = np.array([obj.cls_type for obj in obj_list])
            annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
            annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
            annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
            annotations['yaw'] = np.array([obj.yaw for obj in obj_list])
            annotations['pitch'] = np.array([obj.pitch for obj in obj_list])
            annotations['roll'] = np.array([obj.roll for obj in obj_list])

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

            # # ----------------------------
            # if sample_idx % 200 == 0:
            #     print(self.sample_id_list[sample_idx])
            #     points = self.get_lidar(sample_idx)
            #     pcv = PCViewer(points, gt_boxes_lidar)
            #     pcv.draw_point_cloud()
            # # ----------------------------

            info['annos'] = annotations

            if count_inside_pts:
                points = self.get_lidar(sample_idx)
                corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                for k in range(num_objects):
                    flag = box_utils.in_hull(points[:, 0:3], corners_lidar[k])
                    num_points_in_gt[k] = flag.sum()
                annotations['num_points_in_gt'] = num_points_in_gt

            # # remove empty boxes
            # mask = annotations['num_points_in_gt'] >= 0
            # annotations['num_points_in_gt'] = annotations['num_points_in_gt'][mask]
            # annotations['name'] = annotations['name'][mask]
            # annotations['dimensions'] = annotations['dimensions'][mask]
            # annotations['location'] = annotations['location'][mask]
            # annotations['rotation_y'] = annotations['rotation_y'][mask]
            # annotations['gt_boxes_lidar'] = annotations['gt_boxes_lidar'][mask]
            # annotations['index'] = np.array(list(range(len(annotations['name']))), dtype=np.int32)

        return info

    def transform_box_coords(self, vehicle_from, vehicle_to, boxes):
        rot = vehicle_from['rotation']
        R = transforms3d.euler.euler2mat(rot[0], rot[1], -rot[2]).T
        boxes[:, :3] = np.dot(R, boxes[:, :3].T).T

        boxes[:, :3] += vehicle_from['location'] - vehicle_to['location']

        rot = vehicle_to['rotation']
        R = transforms3d.euler.euler2mat(rot[0], rot[1], rot[2]).T
        boxes[:, :3] = np.dot(R, boxes[:, :3].T).T

        boxes[:, 6] += vehicle_from['rotation'][2] - vehicle_to['rotation'][2]

        return boxes

    def __getitem__(self, index):
        data_dict = super().__getitem__(index)

        if self.dataset_cfg['SAVE_DETECTIONS']:
            return data_dict

        vehicle, frame_num = data_dict['frame'].split('/')
        frame_num = int(frame_num) - self.first_frames_to_delete

        def vehicle_to_idx(v):
            return int(v.split('_')[1])


        coop_boxes = None
        for coop_vehicles in self.dataset_cfg.DATA_SPLIT['all']:
            if coop_vehicles == vehicle:
                continue
            coop_vehicle_idx = vehicle_to_idx(coop_vehicles)
            boxes = self.coop_detections[coop_vehicle_idx][frame_num]['boxes_lidar']
            scores = self.coop_detections[coop_vehicle_idx][frame_num]['score']
            names = self.coop_detections[coop_vehicle_idx][frame_num]['name']
            vehicle_ids = np.full(len(names), coop_vehicle_idx)

            ego_state = self.vehicle_states[vehicle_to_idx(vehicle)][frame_num]
            coop_state = self.vehicle_states[coop_vehicle_idx][frame_num]

            boxes = self.transform_box_coords(coop_state, ego_state, boxes)


            # remove detections of ego vehicle
            min_dist = 1
            dist = np.linalg.norm(boxes[:, :2], axis=1)
            boxes = boxes[dist > min_dist]
            scores = scores[dist > min_dist]
            names = names[dist > min_dist]
            vehicle_ids = vehicle_ids[dist > min_dist]

            # remove gt of ego vehicle
            dist = np.linalg.norm(data_dict['gt_boxes'][:, :2], axis=1)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][dist > min_dist]

            # # todo something with empty coop
            # # remove detections out of range
            # mask = box_utils.mask_boxes_outside_range_numpy(
            #         boxes, self.point_cloud_range,
            #         min_num_corners=1,
            #         use_center_to_filter=True
            #     )
            # boxes = boxes[mask]
            # scores = scores[mask]
            # names = names[mask]
            # vehicle_ids = vehicle_ids[mask]


            if coop_boxes is None:
                coop_boxes = boxes
                coop_scores = scores
                coop_names = names
                coop_vehicle_ids = vehicle_ids
            else:
                coop_boxes = np.concatenate((coop_boxes, boxes), axis=0)
                coop_scores = np.append(coop_scores, scores)
                coop_names = np.append(coop_names, names)
                coop_vehicle_ids = np.append(coop_vehicle_ids, vehicle_ids)


        data_dict['coop_boxes'] = coop_boxes
        data_dict['coop_scores'] = coop_scores
        data_dict['coop_ids'] = np.vectorize(cls_type_to_id)(coop_names)
        data_dict['coop_vehicle_ids'] = coop_vehicle_ids

        #point decoration todo: add multi class support
        if self.dataset_cfg.get('USE_COOP_POINT_DECORATION', False):
            points = data_dict['points']
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                        torch.from_numpy(points[:, :3]),
                        torch.from_numpy(coop_boxes)
                    ).numpy()
            for indices, score in zip(point_indices, coop_scores):
                idx_mask = np.ma.make_mask(indices)
                points[idx_mask, 4] += score
            data_dict['points'] = points


        # only for testin matching and fusion
        # ego_detections = np.concatenate((data_dict['gt_boxes'][:, :7], np.zeros((np.shape(data_dict['gt_boxes'])[0], 1))), axis=1)
        # coop_detections = np.concatenate((coop_boxes, coop_scores.reshape(-1, 1)), axis=1)
        # matches = self.detection_matching(ego_detections, coop_detections, coop_vehicle_ids)
        # final_detections = self.weighted_mean_fusion(matches)

        if self.augmentor is not None:
            data_dict = self.augmentor.forward(data_dict)

        return data_dict

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=None
        )
        self.split = self.dataset_cfg.DATA_SPLIT[split]
        self.sample_id_list = []
        for vehicle in self.split:
            self.sample_id_list += [vehicle + '/' + os.path.basename(x).split('.')[0] for x in sorted(glob.glob(str(self.root_path / vehicle / "lidar/*bin")))][self.first_frames_to_delete:]

    @staticmethod
    def get_objects_from_label(label_file):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        objects = [Object3d(line) for line in lines]
        return objects


def create_pkl_file(dataset_cfg, class_names, data_path, save_path):
    import pickle
    print('--------------- Start to collect infos ---------------')
    dataset = ScopeLidarDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split, all = 'train', 'test', 'all'

    train_filename = save_path / ('scope_infos_%s.pkl' % train_split)
    val_filename = save_path / ('scope_infos_%s.pkl' % val_split)
    all_filename = save_path / ('scope_infos_%s.pkl' % all)

    print('--------------- train infos ---------------')
    dataset.set_split(train_split)
    scope_infos_train = dataset.get_infos(has_label=True, count_inside_pts=True)
    print('train samples:', len(scope_infos_train))
    with open(train_filename, 'wb') as f:
        pickle.dump(scope_infos_train, f)
    print('Scope info train file is saved to %s' % train_filename)

    print('--------------- val infos ---------------')
    dataset.set_split(val_split)
    scope_infos_val = dataset.get_infos(has_label=True, count_inside_pts=True)
    print('val samples:', len(scope_infos_val))
    with open(val_filename, 'wb') as f:
        pickle.dump(scope_infos_val, f)
    print('Scope info val file is saved to %s' % val_filename)

    print('--------------- all infos ---------------')
    dataset.set_split(all)
    scope_infos_all = dataset.get_infos(has_label=True, count_inside_pts=True)
    print('val samples:', len(scope_infos_all))
    with open(all_filename, 'wb') as f:
        pickle.dump(scope_infos_all, f)
    print('Scope info all file is saved to %s' % val_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    #dataset.create_groundtruth_database(save_path, train_filename, split=train_split) # todo
    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import yaml
    from pathlib import Path
    from easydict import EasyDict
    # dataset_path = Path('/home/sven/projects/hannah/datasets/scope/sensor_data_v3/')
    dataset_path = Path('/home/sven/projects/hannah/datasets/scope/test_data_run0/')
    # dataset_path = Path('/nfs/resist/Carla/carla_output_data/town_04_multi_vehicle/test_data_run0/')

    dataset_config = EasyDict(yaml.safe_load(open('/home/sven/projects/hannah/hannah/conf/dataset/scope_lidar.yaml')))
    create_pkl_file(
            dataset_cfg=dataset_config,
            class_names=['Car'],
            data_path=dataset_path,
            save_path=dataset_path)
