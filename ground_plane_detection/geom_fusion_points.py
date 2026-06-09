"""Multi-camera ground-plane detection fusion (graph connect + merge)."""
import math

import networkx as nx
import numpy as np


def connect_detections(detections, R):
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
    """
    graph = nx.Graph()

    for i, di in enumerate(detections):
        for j, dj in enumerate(detections):
            if i < j and di[2] != dj[2]:  # make sure they come from different cameras
                r = np.linalg.norm(np.array(di[:2]) - np.array(dj[:2]))
                if r <= R:
                    graph.add_node(di)
                    graph.add_node(dj)
                    graph.add_edge(di, dj, weight=r)

    return graph


def fusion_multi_camera_detections(detections, graph):
    # Sort detections according to their X coordinates
    sorted_detections = sorted(detections, key=lambda x: x[0])
    # Initialize list to track detections already combined
    found = []
    # Initialize list to store fusion results
    fusion_results = []
    unconnected_detections = []

    # Loop through each detection in sorted order
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
                        # If the neighbor has not already been combined, and it is not from the same camera as the detections in the fusion results list
                        # and it is a neighbor of any detection in the fusion results list
                        if d0 not in found and all(d0[2] != p[2] for p in points) and any(d0 in graph.neighbors(p) for p in points):
                            # Compute the distance between the neighbor and the current detection
                            r = compute_distance(d0, d)
                            rs_dprimes.append((r, d0))

                    # sort the list of tuples by the first element of the tuple (the distance) in ascending order 
                    if len(rs_dprimes) > 0:
                        rs_dprimes_sorted = sorted(rs_dprimes, key=lambda x: x[0])
                        d0 = rs_dprimes_sorted[0][1]

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


def compute_distance(d1, d2):
    # Compute the Euclidean distance between two detections
    return math.sqrt((d1[0] - d2[0])**2 + (d1[1] - d2[1])**2)

def arith_mean(points):
    # Compute the arithmetic mean of a list of detections
    x = sum(p[0] for p in points) / len(points)
    y = sum(p[1] for p in points) / len(points)
    return x, y
