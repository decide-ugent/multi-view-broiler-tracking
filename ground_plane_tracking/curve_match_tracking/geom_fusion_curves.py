import networkx as nx
import scipy.interpolate
import scipy.signal
import numpy as np
import similaritymeasures
from tqdm import tqdm
import pandas as pd

def calculate_lenght_start_end(detection):
    return np.linalg.norm(np.array(detection.list_of_points[0]) - np.array(detection.list_of_points[-1]))
        
def calculate_curve_length(points):
    total_length = 0
    for i in range(1, len(points)):
        total_length += np.linalg.norm(np.array(points[i]) - np.array(points[i-1]))
    return total_length

def compute_distance(d1, d2, intersection=None):
    # Compute the Euclidean distance between two detections
    # return math.sqrt((d1[0] - d2[0])**2 + (d1[1] - d2[1])**2)
    # return np.linalg.norm(np.array(d1.list_of_points[0]) - np.array(d2.list_of_points[0]))

    if intersection:
        points_curve1 = np.array([d1.list_of_points[d1.list_of_frames.index(frame)] for frame in intersection])
        points_curve2 = np.array([d2.list_of_points[d2.list_of_frames.index(frame)] for frame in intersection])
    else:
        points_curve1 = np.array(d1.list_of_points)
        points_curve2 = np.array(d2.list_of_points)

    r = similaritymeasures.frechet_dist(points_curve1, points_curve2)
    # r, _ = similaritymeasures.dtw(points_curve1, points_curve2)
    # r = similaritymeasures.curve_length_measure(points_curve1, points_curve2)

    return r


def arith_mean(points):
    # Compute the arithmetic mean of a list of detections
    x = sum(p[0] for p in points) / len(points)
    y = sum(p[1] for p in points) / len(points)
    return x, y


def connect_detections(detections, R, eucl_R, max_gap=25):
    """
    Connects ground plane detections that are close enough
    in Euclidean distance.

    Parameters:
    - detections: a list of ground plane detections, where each detection
                  is a tuple (x, y, camera_id)
    - R: the maximum Euclidean distance between two detections to be connected

    Returns:
    - a networkx graph, where each node represents a detection, and each edge
      connects two detections that are close enough
    
      
    Detections are now curves 
    """

    graph = nx.Graph()
    # Loop over all pairs of detections
    for i, di in tqdm(enumerate(detections), total=len(detections)):
        for j, dj in enumerate(detections):
            # if i < j and di[2] != dj[2]:  # make sure they come from different cameras
            if i < j: 
                r = 1000000000
                list_of_frames_curve1 = di.list_of_frames
                list_of_frames_curve2 = dj.list_of_frames
                # find the intersection of the two lists
                intersection = list(set(list_of_frames_curve1) & set(list_of_frames_curve2))
                earlier_curve = di if di.list_of_frames[0] < dj.list_of_frames[0] else dj
                later_curve = dj if di.list_of_frames[0] < dj.list_of_frames[0] else di
                last_frame_earlier_curve = earlier_curve.list_of_frames[-1]
                first_frame_later_curve = later_curve.list_of_frames[0]
                diff_frames = first_frame_later_curve - last_frame_earlier_curve

                if di.cam_id != dj.cam_id:
                    if len(intersection) != 0:
                        # take a random point from the intersection and get the points for each curve for that frame
                        random_frame = np.random.choice(intersection)
                        random_pt_curve1 = di.list_of_points[di.list_of_frames.index(random_frame)]
                        random_pt_curve2 = dj.list_of_points[dj.list_of_frames.index(random_frame)]
                        r_eucl = np.linalg.norm(np.array(random_pt_curve1) - np.array(random_pt_curve2))
                        if r_eucl < eucl_R:
                            # continue
                            r = compute_distance(di, dj, intersection)

                    elif diff_frames <= max_gap:
                        r = np.linalg.norm(np.array(earlier_curve.list_of_points[-1]) - np.array(later_curve.list_of_points[0]))
                        # continue
                else:
                    if len(intersection) != 0:
                        continue
                    elif diff_frames <= max_gap:
                        r = np.linalg.norm(np.array(earlier_curve.list_of_points[-1]) - np.array(later_curve.list_of_points[0]))

                if r <= R:
                    graph.add_node(di)
                    graph.add_node(dj)
                    graph.add_edge(di, dj, weight=r)

    return graph, []


def fusion_multi_camera_detections(detections, graph):
    detections_sorted_by_x = sorted(detections, 
                                  key=lambda x: x.list_of_points[0][0])
    
    # Then sort by number of frames (descending) as primary sort key
    # Python's sort is stable, so equal frame counts will maintain x-coordinate ordering
    sorted_detections = sorted(detections_sorted_by_x,
                             key=lambda x: len(x.list_of_frames),
                             reverse=True)

    # Initialize list to track detections already combined
    found = []
    # Initialize list to store fusion results
    fusion_results = []
    unconnected_detections = []
    # Loop through each detection in sorted order
    # for d in tqdm(sorted_detections, total=len(sorted_detections)):
    for d in sorted_detections:
        # If the detection has not already been combined
        points = []
        if d not in found:
            found.append(d)
            points.append(d)
            # Keep looping until no more detections can be added to the fusion results list
            while True:
                # Loop through each neighbor of the current detection
                if d in graph.nodes():
                    rs_dprimes = []
                    for d0 in graph.neighbors(d):
                        if d0 not in found and any(d0 in graph.neighbors(p) for p in points):
                            d0_frames = set(d0.list_of_frames)  # Calculate set once
                            if not any(bool(d0.cam_id == p.cam_id) and bool(d0_frames & set(p.list_of_frames)) for p in points):
                                intersection = list(set(d.list_of_frames) & d0_frames)
                                rs_dprimes.append((graph[d][d0]['weight'], len(intersection), d0))

                    # sort the list of tuples by the first element of the tuple (the distance) in ascending order 
                    if len(rs_dprimes) > 0:
                        # Group results by intersection length
                        intersection_groups = {}
                        for r, intersection_len, d_prime in rs_dprimes:
                            if intersection_len not in intersection_groups:
                                intersection_groups[intersection_len] = []
                            intersection_groups[intersection_len].append((r, intersection_len, d_prime))
                        
                        # Get group with maximum intersection length
                        max_intersection = max(intersection_groups.keys())
                        # From that group, get the one with minimum Frechet distance
                        rs_dprimes_sorted = sorted(intersection_groups[max_intersection], key=lambda x: x[0])

                        d0 = rs_dprimes_sorted[0][2]

                        found.append(d0)
                        points.append(d0)
                        d = d0
                    else:
                        fusion_results.append(points)
                        points = []
                        break
                else:
                    unconnected_detections.append(d)
                    break

    return fusion_results, unconnected_detections


def generate_preds_df(fusion_results, unconnected_detections):
    # print('Fusing curves & adding unconnected detections')
    fusion_id_number = 0
    preds_data = []
    total_number_frames_to_interpolate = 0
    for fusion_result in fusion_results:
        fusion_id_number += 1

        # Convert to integers using int()
        first_frame = int(min([curve.list_of_frames[0] for curve in fusion_result]))
        last_frame = int(max([curve.list_of_frames[-1] for curve in fusion_result]))

        frame_data = []
        for frame in range(first_frame, last_frame + 1):
            # get the points in the curve for the current frame
            to_fuse = []
            for curve in fusion_result:
                if frame in curve.list_of_frames:
                    to_fuse.append(curve.list_of_points[curve.list_of_frames.index(frame)])

            if len(to_fuse) > 0:
                x, y = arith_mean(to_fuse)
                # preds_data.append([int(frame), fusion_id_number, x, y, 1, 1, -1, -1, -1])  # Using -1 as ID for untracked detections
                frame_data.append([int(frame), x, y])
            else:
                # frame_data.append([int(frame), -1, -1])
                total_number_frames_to_interpolate += 1

        for frame, x, y in frame_data:
            preds_data.append([int(frame), fusion_id_number, x, y, 1, 1, -1, -1, -1])  # Using -1 as ID for untracked detections

    # print('Total number of frames to interpolate:', total_number_frames_to_interpolate)
    # print('Adding unconnected detections')
    # for unconn in tqdm(unconnected_detections):
    for unconn in unconnected_detections:
        fusion_id_number += 1
        for frame, point in zip(unconn.list_of_frames, unconn.list_of_points):
            preds_data.append([int(frame), fusion_id_number, int(point[0]), int(point[1]), 1, 1,-1, -1, -1])  # Using -1 as ID for untracked detections
    return preds_data

