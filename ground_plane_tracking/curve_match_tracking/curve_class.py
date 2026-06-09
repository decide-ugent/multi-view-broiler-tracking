import numpy as np
import colorhash

class Curve:
    def __init__(self, list_of_points, list_of_frames, list_of_boxes, cam_id, tracklet_id, list_of_distances):
        self.list_of_points = list_of_points
        self.list_of_frames = list_of_frames
        self.list_of_boxes = list_of_boxes
        self.list_of_distances = list_of_distances
        self.cam_id = cam_id
        self.tracklet_id = tracklet_id
        self.uid = self.gen_uid()

    def get_curve(self):
        return np.array(self.list_of_points, dtype=np.int32)

    def get_color(self):
        return colorhash.ColorHash(self.tracklet_id).rgb

    def get_frames(self):
        return self.list_of_frames
    
    def get_average_curve_distance(self):
        return np.mean(self.list_of_distances)
    
    def multiply_points(self, scale_factor):
        # self.list_of_points = [tuple(np.array(point) * scale_factor) for point in self.list_of_points]
        self.list_of_points = [(int(point[0] * scale_factor), int(point[1] * scale_factor)) for point in self.list_of_points]


    def gen_uid(self):
        return f'{self.cam_id}_{self.tracklet_id}'