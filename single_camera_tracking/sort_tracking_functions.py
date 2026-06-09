from sort.sort import *

def load_boxes(labelpath, threshold=0.5):
    classes = []
    boxes = [] 
    confs = []
    processed_boxes = []

    # Load detections
    with open(labelpath, 'r') as f:
        lines = f.readlines()

        for line in lines:
            line = line.split()
            cls = int(line[0])
            box = list(map(float, line[1:5]))
            conf = float(line[-1])

            classes.append(cls)
            boxes.append(box)
            confs.append(conf)

            if conf > threshold:
                x, y, w, h = box
                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)
                processed_boxes.append([x1, y1, x2, y2])

    return classes, boxes, confs, np.array(processed_boxes)

def track_video_sort(all_dets, param_dict, debug=False):
    mot_tracker = Sort(max_age=param_dict['MAX_AGE'], min_hits=param_dict['MIN_HITS'], iou_threshold=param_dict['IOU_THRESHOLD']) 
    tracklets = []

    for frame_index, yolo_det_txt in all_dets.items():
        if debug and frame_index > 15:
            break
        classes, boxes, confs, processed_boxes = load_boxes(yolo_det_txt, threshold=param_dict['BB_THRESHOLD'])

        # Check if there are no boxes detected in the current frame
        if len(processed_boxes) == 0:
            # If no boxes are detected, use an empty array for the tracker update.
            print('No boxes detected in frame', frame_index)
            processed_boxes = np.empty((0, 5))

        track_bbs_ids = np.empty((0, 5))
        track_bbs_ids = mot_tracker.update(processed_boxes)

        total_tracks_found = len(track_bbs_ids.tolist())
        # print(total_tracks_found, len(processed_boxes), 'unused_boxes', len(processed_boxes) - total_tracks_found)

        for j in range(0, total_tracks_found):
            coords = track_bbs_ids.tolist()[j]
            x1, y1, x2, y2, name_idx = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]), int(coords[4])
            bb_height = y2 - y1
            bb_width = x2 - x1

            tracklets.append([frame_index, name_idx, x1, y1, bb_width, bb_height, -1, -1, -1])
    return tracklets