import numpy as np
from pcdet.utils import box_utils
import open3d
import matplotlib.pyplot as plt

class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.type = label[0]
        self.truncation = float(label[1])
        self.occlusion = float(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.loc)
        self.ry = float(label[14])
        self.score = float(label[15]) if label.__len__() == 16 else -1.0

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.loc
        return corners3d

    def cam_to_velo(self, calib):
        velo_to_cam = np.append(calib['Tr_velo2cam'], np.array([[0, 0, 0, 1]]), axis=0) # 4x4 homogeneous
        cam_to_velo = np.linalg.inv(velo_to_cam) # 4x4 homogeneous
        corner_points_cam = self.generate_corners3d() # 3x8
        corner_points_cam = np.hstack((corner_points_cam, np.ones((corner_points_cam.shape[0], 1), dtype=np.float32))) # 3x8 homogeneous
        return np.matmul(corner_points_cam, np.transpose(cam_to_velo))[:, 0:3] # corner_points_cam * cam_to_velo^T

    def get_lidar_bbox(self, calib):
        return self.cam_to_velo(calib)


class PCViewer:
    def __init__(self, point_cloud=None, gt_boxes=None, pred_boxes=None):
        self.point_cloud = point_cloud
        self.gt_boxes = gt_boxes
        self.pred_boxes = pred_boxes

    def set_point_cloud(self, point_cloud):
        self.point_cloud = point_cloud

    def set_gt_boxes(self, gt_boxes):
        self.gt_boxes = gt_boxes

    def set_pred_boxes(self, pred_boxes, pred_scores, min_score=0.5):
        to_delete = []
        for i, score in enumerate(pred_scores):
            if score < min_score:
                to_delete.append(i)
        self.pred_boxes = np.delete(pred_boxes, to_delete, axis=0)

    def draw_point_cloud(self):
        points = self.point_cloud[:, :3]
        intensities = 1-(self.point_cloud[:, 3:].reshape(-1))
        cm = plt.get_cmap('jet')
        colors = np.array(cm(intensities))[:, :3]
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points)
        pcd.colors = open3d.utility.Vector3dVector(colors)
        vis = open3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        opt.point_size = 0.5

        lines = [[0, 1], [1, 2], [2, 3], [0, 3],
                 [4, 5], [5, 6], [6, 7], [4, 7],
                 [0, 4], [1, 5], [2, 6], [3, 7]]

        if self.gt_boxes is not None:
            corner_boxes = box_utils.boxes_to_corners_3d(self.gt_boxes)
            colors = [[0, 1, 0] for _ in range(len(lines))]
            for gt_box, corner_box in zip(self.gt_boxes, corner_boxes):
                if gt_box[0] < -900:
                    continue
                line_set = open3d.geometry.LineSet()
                line_set.points = open3d.utility.Vector3dVector(corner_box)
                line_set.lines = open3d.utility.Vector2iVector(lines)
                line_set.colors = open3d.utility.Vector3dVector(colors)
                vis.add_geometry(line_set)

        if self.pred_boxes is not None:
            corner_boxes = box_utils.boxes_to_corners_3d(self.pred_boxes)
            colors = [[1, 0, 0] for _ in range(len(lines))]
            for corner_box in corner_boxes:
                line_set = open3d.geometry.LineSet()
                line_set.points = open3d.utility.Vector3dVector(corner_box)
                line_set.lines = open3d.utility.Vector2iVector(lines)
                line_set.colors = open3d.utility.Vector3dVector(colors)
                vis.add_geometry(line_set)

        vis.run()
        vis.destroy_window()


