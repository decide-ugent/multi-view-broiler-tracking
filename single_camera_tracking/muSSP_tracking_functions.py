from tqdm import tqdm
import mot3d
import mot3d.weight_functions as wf

from single_camera_tracking.sort_tracking_functions import load_boxes

def muSSP_tracking(all_dets, start, stop,
                                  sigma_distance=10, 
                                  max_distance=30, 
                                  sigma_motion=50,
                                  max_jump_pass1=4, 
                                  max_jump_pass2=20,
                                  weight_source_sink=0.1,
                                  length=5,
                                  th_length=2,
                                  sigma_box_size=0.3,
                                  BB_THRESHOLD=0.45,
                                  verbose=True):

    detections = []
    for index in tqdm(range(start, stop)):
        bb_box_path = all_dets[index]
        classes, boxes, confs, processed_boxes = load_boxes(bb_box_path, threshold=BB_THRESHOLD)
        for bb_box in processed_boxes: # first index is zero which is the first frame with id 000001.txt
            # id = position[2]
            # id = 'notused'
            x1, y1, x2, y2 = bb_box
            center = [(x1+x2)/2, (y1+y2)/2]
            # print(center)
            # position = [position[0], position[1]]
            bounding_box = (x1, y1, x2, y2)

            detections.append(mot3d.Detection2D(index, center,
                                                # color_histogram=dummy_color_histogram,
                                                bbox=bounding_box))#, id=id))

    if verbose:
        print("Building the graph..")

    weight_distance = lambda d1, d2: wf.weight_distance_detections_2d(d1, d2,
                                                                      sigma_jump=1, sigma_distance=sigma_distance,sigma_box_size=sigma_box_size,
                                                                    #   sigma_color_histogram=0.3, ,
                                                                      max_distance=max_distance,
                                                                    #   use_color_histogram=True, use_bbox=True)
                                                                      use_color_histogram=False, use_bbox=True)
    
    weight_confidence = lambda d: wf.weight_confidence_detections_2d(d, mul=1, bias=0)

    g = mot3d.build_graph(detections, weight_source_sink=weight_source_sink,
                          max_jump=max_jump_pass1, verbose=verbose,
                          weight_confidence=weight_confidence,
                          weight_distance=weight_distance)
    if g is None:
        # raise RuntimeError("There is not a single path between sink and source nodes!") 
        print("There is not a single path between sink and source nodes!") 
        return []
    if verbose:
        print("-"*30)
        print("Solving the graph with muSSP..")
        print("-"*30)


    trajectories = mot3d.solve_graph(g, verbose=verbose, method='muSSP')
    tracklets = []
    for traj in trajectories:
        tracklets += mot3d.split_trajectory_modulo(traj, length=length)

    tracklets = mot3d.remove_short_trajectories(tracklets, th_length=th_length)
    if verbose:
        print("Building graph on tracklets..")
    detections_tracklets = [mot3d.DetectionTracklet2D(tracklet) for tracklet in tracklets]
    
    weight_distance_t = lambda t1, t2: wf.weight_distance_tracklets_2d(t1, t2, max_distance=None, sigma_motion=sigma_motion,
                                                                    #    sigma_color_histogram=0.3, ,# alpha=0.7,
                                                                       cutoff_motion=0.1,# cutoff_appearance=0.1, # sigma_color_histogram, alpha and cutoff are not used when use_color_histogram=False
                                                                       use_color_histogram=False)
    
    weight_confidence_t = lambda t: wf.weight_confidence_tracklets_2d(t, mul=1, bias=0)

    g = mot3d.build_graph(detections_tracklets, weight_source_sink=weight_source_sink,
                          max_jump=max_jump_pass2, verbose=verbose,
                          weight_confidence=weight_confidence_t,
                          weight_distance=weight_distance_t)    
    if g is None:
        # raise RuntimeError("There is not a single path between sink and source nodes!") 
        print("There is not a single path between sink and source nodes!") 
        return []
    if verbose:
        print("-"*30)
        print("Solver second pass on tracklets..")
        print("-"*30)
    trajectories = mot3d.solve_graph(g, verbose=verbose, method='muSSP')    
    if verbose:
        print("number of trajectories", len(trajectories))

    mussp_output = []
    id_ = 0
    for traj in trajectories:
        for detection in traj:
            for det in detection.tracklet:
                x1, y1, x2, y2 = det.bbox
                # mussp_output.append([int(det.index), id_, det.position[0], det.position[1], 1, 1, -1, -1, -1])
                bb_height = y2 - y1
                bb_width = x2 - x1 
                mussp_output.append([int(det.index), id_, x1, y1, bb_width, bb_height, -1, -1, -1])
        id_ += 1

    return mussp_output