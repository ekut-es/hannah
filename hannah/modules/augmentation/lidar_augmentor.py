import numpy as np
from scipy.stats import truncnorm
from scipy.spatial.transform import Rotation
from . import raytracing
from pcdet.utils import box_utils
import torch
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
#from pointcloud_viewer import PCViewer


class LidarAugmentor:
    def __init__(self, **kwargs):
        self.point_cloud = None
        self.labels = None
        self.coop_boxes = None
        self.names = None
        self.augmentations = kwargs['augmentations']

    @staticmethod
    def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    def random_noise(self, ranges, sigma, type, enabled):
        if not enabled:
            return
        r = ranges
        num_points = abs(int(np.random.normal(loc=0.0, scale=sigma)))
        x = np.random.uniform(r[0], r[1], num_points)
        y = np.random.uniform(r[2], r[3], num_points)
        z = np.random.uniform(r[4], r[5], num_points)
        if type == 'uniform':
            i = np.random.uniform(r[6], r[7], num_points)
        elif type == 'salt_pepper':
            salt = np.zeros(int(num_points/2))
            pepper = np.ones(num_points-int(num_points/2)) # todo 255
            i = np.append(salt, pepper)
        elif type == 'min':
            i = np.zeros(num_points)
        elif type == 'max':
            i = np.full(num_points, 255)
        else:
            raise ValueError('No matching intensity type given')

        noise = np.stack((x, y, z, i), axis=-1)
        self.point_cloud = np.concatenate((self.point_cloud, noise), axis=0)

    def thin_out(self, sigma, enabled):
        if not enabled:
            return
        PercentDistribution = self.get_truncated_normal(mean=0, sd=sigma, low=0, upp=1)
        percent = PercentDistribution.rvs()
        idx = np.random.choice(self.point_cloud.shape[0], int(np.ceil(len(self.point_cloud)*(1-percent))), replace=False)
        self.point_cloud = self.point_cloud[idx, :]

    def random_translation(self, sigma, enabled):
        if not enabled:
            return
        x_translation = np.random.normal(loc=0.0, scale=sigma)
        y_translation = np.random.normal(loc=0.0, scale=sigma)
        z_translation = np.random.normal(loc=0.0, scale=sigma)

        self.point_cloud[:, 0] += x_translation
        self.point_cloud[:, 1] += y_translation
        self.point_cloud[:, 2] += z_translation

        self.labels[:, 0] += x_translation
        self.labels[:, 1] += y_translation
        self.labels[:, 2] += z_translation

        self.coop_boxes[:, 0] += x_translation
        self.coop_boxes[:, 1] += y_translation
        self.coop_boxes[:, 2] += z_translation

    def random_scaling(self, sigma, max_scale, enabled):
        if not enabled:
            return

        scale_dist = self.get_truncated_normal(mean=1, sd=sigma, low=(1/max_scale), upp=max_scale)
        scale_factor = scale_dist.rvs()

        self.point_cloud[:, :3] *= scale_factor
        self.labels[:, :6] *= scale_factor
        self.coop_boxes[:, :6] *= scale_factor

    def local_scaling(self, sigma, enabled, max_scale):
        if not enabled:
            return

        scale_dist = self.get_truncated_normal(mean=1, sd=sigma, low=(1 / max_scale), upp=max_scale)
        scale_factor = scale_dist.rvs()

        point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
            torch.from_numpy(self.point_cloud[:, 0:3]), torch.from_numpy(self.labels)
        ).numpy()

        for box, points in zip(self.labels, point_indices):
            points = np.ma.make_mask(points)
            if not np.any(points):
                continue
            self.point_cloud[points, :3] -= box[:3]
            self.point_cloud[points, :3] *= scale_factor
            self.point_cloud[points, :3] += box[:3]
        self.labels[:, 3:6] *= scale_factor

    def random_rotation(self, sigma, enabled):
        if not enabled:
            return

        rot_dist = self.get_truncated_normal(mean=0, sd=sigma, low=-180, upp=180)
        rot_angle = rot_dist.rvs()

        rotation = Rotation.from_euler('z', rot_angle, degrees=True)
        self.point_cloud[:, :3] = rotation.apply(self.point_cloud[:, :3])
        self.labels[:, :3] = rotation.apply(self.labels[:, :3])
        self.labels[:, 6] = (self.labels[:, 6] + np.radians(rot_angle)) % (2*np.pi)
        self.coop_boxes[:, :3] = rotation.apply(self.coop_boxes[:, :3])
        self.coop_boxes[:, 6] = (self.coop_boxes[:, 6] + np.radians(rot_angle)) % (2*np.pi)

    def random_flip(self, prob, enabled):
        if not enabled:
            return
        if prob > np.random.rand():
            self.point_cloud[:, 1] *= -1
            self.labels[:, 1] *= -1
            self.labels[:, 6] = (self.labels[:, 6] + np.pi) % (2 * np.pi)

    def random_intensity_shift(self, sigma, enabled):
        if not enabled:
            return

        shift = int(np.random.normal(loc=0.0, scale=sigma))
        self.point_cloud[:, 3] += shift
        self.point_cloud[self.point_cloud[:, 3] < 0, 3] = 0
        self.point_cloud[self.point_cloud[:, 3] > 255, 3] = 255

    def fog(self, prob, metric, sigma, enabled, mean):
        if not enabled or prob < np.random.rand():
            return

        PercentDistribution = self.get_truncated_normal(mean=int(mean), sd=sigma, low=10, upp=int(mean))
        viewing_dist = PercentDistribution.rvs()

        if metric == 'dist':
            extinction_factor = 0.32 * np.exp(-0.022 * viewing_dist)
            beta = (-0.00846 * viewing_dist) + 2.29
            delete_probability = -0.63 * np.exp(-0.02 * viewing_dist) + 1

        elif metric == 'chamfer':
            extinction_factor = 0.23 * np.exp(-0.0082 * viewing_dist)
            beta = (-0.006 * viewing_dist) + 2.31
            delete_probability = -0.7 * np.exp(-0.024 * viewing_dist) + 1
        else:
            raise ValueError('No matching metric type given')

        # selecting points for modification and deletion
        dist = np.sqrt(np.sum(self.point_cloud[:, :3] ** 2, axis=1))
        modify_probability = 1 - np.exp(-extinction_factor * dist)
        modify_threshold = np.random.rand(len(modify_probability))
        selected = modify_threshold < modify_probability
        delete_threshold = np.random.rand(len(self.point_cloud))
        deleted = np.logical_and(delete_threshold < delete_probability, selected)

        # changing intensity of unaltered points according to beer lambert law
        self.point_cloud[np.logical_not(selected), 3] *= np.exp(-(2.99573 / viewing_dist) * 2 * dist[np.logical_not(selected)])

        # changing position and intensity of selected points
        altered_points = np.logical_and(selected, np.logical_not(deleted))
        num_altered_points = len(self.point_cloud[altered_points, :3])
        if num_altered_points > 0:
            newdist = np.random.exponential(beta, num_altered_points) + 1.3
            self.point_cloud[altered_points, :3] *= np.reshape(newdist / dist[altered_points], (-1, 1))
            self.point_cloud[altered_points, 3] = np.random.uniform(0, 82, num_altered_points)

        # delete points
        self.point_cloud = self.point_cloud[np.logical_not(deleted), :]

    def universal_weather(self, prob, sigma, enabled, mean, ext_a, ext_b, beta_a, beta_b, del_a, del_b, int_a, int_b, mean_int, int_range):
        if not enabled or prob < np.random.rand():
            return

        PercentDistribution = self.get_truncated_normal(mean=mean, sd=sigma, low=0, upp=mean)
        viewing_dist = PercentDistribution.rvs()

        extinction_factor = ext_a * np.exp(ext_b * viewing_dist)
        beta = (-beta_a * viewing_dist) + beta_b
        delete_probability = -del_a * np.exp(-del_b * viewing_dist) + 1

        # selecting points for modification and deletion
        dist = np.sqrt(np.sum(self.point_cloud[:, :3] ** 2, axis=1))
        modify_probability = 1 - np.exp(-extinction_factor * dist)
        modify_threshold = np.random.rand(len(modify_probability))
        selected = modify_threshold < modify_probability
        delete_threshold = np.random.rand(len(self.point_cloud))
        deleted = np.logical_and(delete_threshold < delete_probability, selected)

        # changing intensity of unaltered points according to parametrized beer lambert law
        self.point_cloud[np.logical_not(selected), 3] *= int_a * np.exp(-(int_b / viewing_dist) * dist[np.logical_not(selected)])

        # changing position and intensity of selected points
        altered_points = np.logical_and(selected, np.logical_not(deleted))
        num_altered_points = len(self.point_cloud[altered_points, :3])
        if num_altered_points > 0:
            newdist = np.random.exponential(beta, num_altered_points) + 1.3
            self.point_cloud[altered_points, :3] *= np.reshape(newdist / dist[altered_points], (-1, 1))
            min_int = mean_int - (int_range/2)
            min_int = min_int if min_int >= 0 else 0
            max_int = mean_int + (int_range/2)
            max_int = max_int if max_int <= 255 else 255
            self.point_cloud[altered_points, 3] = np.random.uniform(min_int, max_int, num_altered_points)

        # delete points
        self.point_cloud = self.point_cloud[np.logical_not(deleted), :]
        self.point_cloud[self.point_cloud[:, 3] > 255, 3] = 255

    def rain(self, precipitation_sigma, number_drops_sigma, noise_filter_path, enabled, prob):
        if not enabled or prob < np.random.rand():
            return
        R_dist = self.get_truncated_normal(mean=0, sd=precipitation_sigma, low=0, upp=20)
        R = (np.floor(R_dist.rvs()).astype(np.int32) + 1)
        N_dist = self.get_truncated_normal(mean=0, sd=number_drops_sigma, low=0, upp=6)
        N = (np.floor(N_dist.rvs()).astype(np.int32) + 1) * 200

        noise_file = noise_filter_path + 'nf_N=' + str(N) + '_R=' + str(R) + '.npz'
        noisefilter = np.load(noise_file)
        nf = noisefilter['nf']
        si = noisefilter['si']
        self.point_cloud = raytracing.cuda_trace(self.point_cloud, nf, si, intensity_factor=0.9)

    def snow(self, precipitation_sigma, number_drops_sigma, noise_filter_path, enabled, prob, scale):
        if not enabled or prob < np.random.rand():
            return
        R_dist = self.get_truncated_normal(mean=0, sd=precipitation_sigma, low=0, upp=10)
        R = (np.floor(R_dist.rvs()).astype(np.int32) + 1)
        N_dist = self.get_truncated_normal(mean=0, sd=number_drops_sigma, low=0, upp=12)
        N = (np.floor(N_dist.rvs()).astype(np.int32) + 1) * 100

        noise_file = noise_filter_path + 'nf_N=' + str(N) + '_R=' + str(R) + '.npz'
        noisefilter = np.load(noise_file)
        nf = noisefilter['nf']
        nf[:, 4] *= scale
        si = noisefilter['si']
        self.point_cloud = raytracing.cuda_trace(self.point_cloud, nf, si, intensity_factor=1.25)
        self.point_cloud[self.point_cloud[:, 3] > 255, 3] = 255

    def delete_labels_by_min_points(self, enabled, min_points):
        if not enabled:
            return
        num_labels = len(self.labels)
        num_points_in_gt = -np.ones(num_labels, dtype=np.int32)
        corners_lidar = box_utils.boxes_to_corners_3d(self.labels)
        for i in range(num_labels):
            flag = box_utils.in_hull(self.point_cloud[:, 0:3], corners_lidar[i])
            num_points_in_gt[i] = flag.sum()

        not_deleted = num_points_in_gt >= min_points
        self.labels = self.labels[not_deleted]
        self.names = self.names[not_deleted]

    def forward(self, batch):
        self.point_cloud = batch.get('points')
        self.labels = batch.get('gt_boxes')
        self.names = batch.get('gt_names')
        self.coop_boxes = batch.get('coop_boxes')

        for augmentation, params in self.augmentations.items():
            try:
                self.call(augmentation, params)
            except Exception as e:
                print('encountered exception in lidar augmentor')
                print('exception: ', e)
                print('batch: ', batch)
                print('augmentation: ', augmentation)
                print('params: ', params)

        batch['points'] = self.point_cloud
        batch['gt_boxes'] = self.labels
        batch['gt_names'] = self.names
        batch['coop_boxes'] = self.coop_boxes
        return batch

    def call(self, augmentation, params):
        a = getattr(self, augmentation)
        a(**params)



