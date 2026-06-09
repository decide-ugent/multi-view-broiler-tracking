import numpy as np
import cv2
import os
import json
from ground_plane_detection.MOD_metrics import calculate_MOD_metrics
from ground_plane_detection.geom_fusion_points import connect_detections, fusion_multi_camera_detections, arith_mean
from ground_plane_detection.mv_utils import pythonify, get_combined_view_6_frames, update_topdown_visualization, update_topdown_visualization_with_unconnected, visualize_one_view,get_bbs_from_txt, calculate_dt, save_tps_fps_fns
from utils.projection import project_2d_points_to_3d_with_distortion, distance_of_2d_pixel_to_camera_new

def process_2d_points(ann_path, calib, bb_threshold, option, min_dist, max_dist, need_normalization=True):
    width = 1920
    height = 1080
    bounding_boxes = get_bbs_from_txt(ann_path, width, height, bb_threshold, need_normalization=need_normalization)

    points_to_project = []
    
    if len(bounding_boxes) > 0:
        bounding_boxes = np.array(bounding_boxes)
        xmin, ymin, xmax, ymax = bounding_boxes[:, 0], bounding_boxes[:, 1], bounding_boxes[:, 2], bounding_boxes[:, 3]
        
        # Calculate center x for all options
        center_x = np.int32((xmin + xmax) / 2)
        
        if option == 'center':
            center_y = np.int32((ymin + ymax) / 2)
            points_to_project = np.column_stack((center_x, center_y))
        elif option == 'bottom':
            points_to_project = np.column_stack((center_x, ymax.astype(np.int32)))
        elif option == 'distance':
            # This option requires per-point processing due to the distance calculation
            points_to_project = []
            for i in range(len(bounding_boxes)):
                centerpoint = [int((xmin[i] + xmax[i])/2), int((ymin[i] + ymax[i])/2)]
                center_dist, _, _ = distance_of_2d_pixel_to_camera_new(calib, centerpoint, scalefactor=100)
                factor = (center_dist - min_dist) / (max_dist - min_dist)
                factor = 1 - np.clip(factor, 0, 1)
                y_point = int(ymax[i] - ((ymax[i] - ymin[i])/2) * factor)
                points_to_project.append([int(center_x[i]), y_point])
    
    # Convert to the expected format for the rest of the function
    points_2d = np.array(points_to_project, dtype=np.float64).reshape(-1, 1, 2)

    return points_2d, bounding_boxes

def model(sample, cam_set, calibs, radius, bb_threshold, option, remove_threshold, vis=True, min_dist=1000, max_dist=10000, save_image_path=None, need_normalization=True):
    if vis:
        final_contour = np.array([[ 236 , 66 ],[ 332 , 46 ],[ 1033 , 79 ],[ 1104 , 102 ],[ 1108 , 358 ],[ 921 , 438 ],[ 399 , 462 ],[ 348 , 403 ],[ 259 , 367 ]])
        pen_total_area_height = 500
        pen_total_area_width = 1189
        topdownframe = np.ones((pen_total_area_height, pen_total_area_width, 3), dtype=np.uint8) * 255
        topdownframe_fused = np.ones((pen_total_area_height, pen_total_area_width, 3), dtype=np.uint8) * 255
        four_frames = []

    yolo_set = sample['yolo_set']
    yolo_set = pythonify(yolo_set)
    cam_nums = list(yolo_set.keys())
    points_per_frame = []

    for cam_num in yolo_set:
        if cam_num not in cam_set:
            # print(cam_num, cam_set)
            continue

        ann_path = yolo_set[cam_num]
        points_2d, bounding_boxes = process_2d_points(ann_path, calibs[cam_num], bb_threshold, option, min_dist, max_dist, need_normalization=need_normalization)
        if points_2d.shape[0] == 0:
            continue

        projected_points =  project_2d_points_to_3d_with_distortion(points_2d, calibs[cam_num], scale=100)
        projected_points = np.array(projected_points, dtype=np.float64).reshape(-1, 2)

        cam_loc = calibs[cam_num]['cam_location'][:2] * 100
        distances = np.linalg.norm(projected_points - cam_loc, axis=1)
        valid_points = distances < remove_threshold
        new_projected_points = projected_points[valid_points]
        projected_points = np.array(new_projected_points)

        if projected_points.shape[0] == 0:
            continue

        projected_points_with_cam_num = np.hstack([projected_points, np.ones((projected_points.shape[0], 1)) * cam_num])
        points_per_frame.append(projected_points_with_cam_num)

        if vis:
            topdown_per_view = np.ones((pen_total_area_height, pen_total_area_width, 4), dtype=np.uint8) * 255
            # colors = [(255, 0, 0), (0, 0, 255), (0, 200, 0),(0, 255, 255) ]
            colors_alpha = [(255, 0, 0, 255), (0, 0, 255, 255), (0, 200, 0, 255),(255, 255, 0, 255) ]
            # set the alpha channel to transparent
            topdown_per_view[:, :, 3] = 0
            cam_num_index = cam_nums.index(cam_num)
            color = colors_alpha[cam_num_index]
            in_contour = cv2.pointPolygonTest(final_contour, new_projected_points.astype(int), False) > 0
            cv2.circle(topdown_per_view, new_projected_points[in_contour].astype(int), 7, (0,0,0,255), -1)
            cv2.circle(topdown_per_view, new_projected_points[in_contour].astype(int), 5, color, -1)

            img = cv2.imread(sample['frame_set'][cam_num])
            topdown_cropped = topdown_per_view[50:465, 240:1120, :]
            # # add a black border around the rim 
            # topdown_cropped = cv2.copyMakeBorder(topdown_cropped, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(0, 255, 255, 255))
            os.makedirs(os.path.join(save_image_path, 'demonstration_images'), exist_ok=True)
            save_path = os.path.join(save_image_path, 'demonstration_images', f'cam_{cam_num}_topdown_contoured.png')
            cv2.imwrite(save_path, topdown_cropped)
            img = visualize_one_view(img, bounding_boxes, points_2d, projected_points, cam_num, cam_nums, cam_num_index)

            save_path = os.path.join(save_image_path, 'demonstration_images', f'cam_{cam_num}_img.jpg')
            cv2.imwrite(save_path, img)
            four_frames.append(img)


    if len(points_per_frame) == 0:
        return np.array([]), None, None

    points_per_frame = np.vstack(points_per_frame)
    # print(points_per_frame.shape)(242, 3)
    # every row in points_per_frame is a point with x, y, cam_num and should be a tuple 
    points_per_frame = [tuple(x) for x in points_per_frame]

    g = connect_detections(points_per_frame, radius)
    fusion_results, unconnected_dets = fusion_multi_camera_detections(points_per_frame, g)
    final_fused_detections = []

    # adding connected detections to the result
    for i in range(len(fusion_results)):
        x, y = arith_mean(fusion_results[i])
        final_fused_detections.append((x, y))

        if vis:
            topdownframe, topdownframe_fused = update_topdown_visualization(topdownframe, topdownframe_fused, fusion_results[i], cam_nums, x, y)

    # adding unconnected detections to the result
    for i in range(len(unconnected_dets)):
        x = unconnected_dets[i][0]
        y = unconnected_dets[i][1]
        final_fused_detections.append((x, y))

        if vis:
            topdownframe, topdownframe_fused = update_topdown_visualization_with_unconnected(topdownframe, topdownframe_fused, unconnected_dets, cam_nums, i)

    # remove the points that are outside the pen 
    new_final_fused_detections = []

    # contour_pen_3 = np.array([[1146,  51], [1150 , 452], [ 251,  449], [ 255  , 51]])
    contour_pen_3 = np.array([[1120,  50], [1120 , 465], [ 240,  465], [ 240  , 50]])  # updated contour
    for detection in final_fused_detections:
        if cv2.pointPolygonTest(contour_pen_3, (detection[0], detection[1]), False) > 0:
            new_final_fused_detections.append(detection)
    
    final_fused_detections = new_final_fused_detections

    if vis:
        add_gts = False
        if 'gt_locs' in sample and add_gts:
            gt_locs = np.array(list(sample['gt_locs'].values()))
            for (x, y) in gt_locs:
                cv2.circle(topdownframe_fused, (int(x), int(y)), 5, (255, 255, 255), -1)
                cv2.circle(topdownframe_fused, (int(x), int(y)), 2, (0,0,0), -1)

        four_frames.append(topdownframe_fused)
        # topdownframe = add_landmarks(topdownframe, None, pen_number ,False)
        four_frames.append(topdownframe) # actaully sixt frame

        # topdown_cropped = topdownframe_fused[50:465, 240:1120, :]
        # topdown_cropped = cv2.copyMakeBorder(topdown_cropped, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(0, 255, 255, 255))
        # cv2.imwrite('topdownframe_fused.jpg', topdown_cropped)
        # topdown_cropped = topdownframe[50:465, 240:1120, :]
        # topdown_cropped = cv2.copyMakeBorder(topdown_cropped, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(0, 255, 255, 255))
        # cv2.imwrite('topdownframe.jpg', topdown_cropped)

        combined_frames = get_combined_view_6_frames(four_frames)
        return np.array(final_fused_detections), combined_frames, four_frames

    # return np.array(final_fused_detections), None, None
    return final_fused_detections, None, None
    
def run_one_sample(sample, calibs, mask_dict, param_dict, save_image_path=None, visualize=True):
    """
    Process a single sample and evaluate detection performance.
    
    Args:
        sample: Dictionary containing sample data
        calibs: Camera calibration data
        mask_dict: Masks for each camera view
        save_image_path: Path to save visualization results
        visualize: Whether to generate and save visualization outputs
    
    Returns:
        result_dict: Dictionary with evaluation metrics
        true_positives, false_positives, false_negatives: Lists of detection results
    """
    age = sample['age']
    dt = calculate_dt(age)

    # Call model with visualization flag
    pred_locs, combined_frames, six_frames = model(sample,
                            cam_set=param_dict['cam_set'], # for evaluating performance of specific camera sets
                            calibs=calibs, 
                            radius=dt*2,  
                            bb_threshold=param_dict['bb_threshold'],
                            option=param_dict['option'], 
                            remove_threshold=param_dict['remove_threshold'], 
                            vis=visualize, 
                            min_dist=param_dict['min_dist'], 
                            max_dist=param_dict['max_dist'],
                            save_image_path=save_image_path
                            )
    pred_locs = np.array(pred_locs).astype(int)

    gt_locs = sample['gt_locs']
    gt_locs = np.array(list(gt_locs.values()))

    # Calculate metrics (needed regardless of visualization)
    if pred_locs.shape[0] == 0:
        recall = 0
        precision = 0
        moda = 0
        modp = 0
        true_positives = []
        false_positives = []
        false_negatives = []
    else:
        detRaw = np.hstack([np.zeros((pred_locs.shape[0], 1)), pred_locs])
        gtRaw = np.hstack([np.zeros((gt_locs.shape[0], 1)), gt_locs])
        recall, precision, moda, modp, distances = calculate_MOD_metrics(gtRaw, detRaw, td=dt)

        true_positives, false_positives, false_negatives = save_tps_fps_fns(gt_locs, pred_locs, distances, dt)

    result_dict  = {'recall': recall, 
                    'precision': precision, 
                    'moda': moda, 
                    'modp': modp, 
                    'true_positives': true_positives,
                    'false_positives': false_positives,
                    'false_negatives': false_negatives,
                    'timestamp': sample['date'], 
                    'dt': dt, 
                    'date': sample['date'], 
                    'category': sample['phase'],
                    'detections': len(pred_locs), 
                    'gt': len(gt_locs)}
    
    # Only perform visualization if requested and save_image_path is provided
    if visualize and save_image_path is not None:
        # Create directory for qualitative results
        json_root_path = os.path.join(save_image_path, 'qualitative')
        if not os.path.exists(json_root_path):
            os.makedirs(json_root_path)

        # Save predictions and ground truth in JSON format
        json_save_path = 'sample_' + str(sample['sample_number']) + '.json'
        json_save_path = os.path.join(json_root_path, json_save_path)

        locations = {
            'preds': pred_locs.tolist(),
            'gts': gt_locs.tolist()
        }
        with open(json_save_path, 'w') as f:
            json.dump(locations, f)

        # Generate and save visualizations for each camera view
        yolo_set = sample['yolo_set']
        cam_nums = list(yolo_set.keys())

        for i in range(4):
            im_name = 'sample_' + str(sample['sample_number']) + '_cam_' + str(cam_nums[i]) + '.jpg'

            cam_img = six_frames[i]
            msize = 15
            thick = 2

            # Reproject predicted points to camera view
            calib = calibs[cam_nums[i]]

            for pred_loc in pred_locs:
                x = pred_loc[0]
                y = pred_loc[1]

                if mask_dict[cam_nums[i]][y, x] == 0:
                    continue

                x = x/100
                y = y/100

                worldgrid = np.array([x, y, 0], dtype=float).reshape(3, 1)
                imgpts = cv2.projectPoints(worldgrid, calib['rvec'], calib['tvec'], calib['cameraMatrix'], calib['distCoeffs'])[0][0][0]
                xim, yim = int(imgpts[0]), int(imgpts[1])
                if xim < 0 or xim > 1920 or yim < 0 or yim > 1080:
                    continue
                cv2.drawMarker(cam_img, (xim, yim), (0, 255, 0), markerType=cv2.MARKER_TRIANGLE_UP, markerSize=msize, thickness=thick, line_type=cv2.LINE_AA)

            # Reproject ground truth points to camera view
            for gt_loc in gt_locs:
                x = gt_loc[0]
                y = gt_loc[1]

                if mask_dict[cam_nums[i]][y, x] == 0:
                    continue

                x = x/100
                y = y/100

                worldgrid = np.array([x, y, 0], dtype=float).reshape(3, 1)
                imgpts = cv2.projectPoints(worldgrid, calib['rvec'], calib['tvec'], calib['cameraMatrix'], calib['distCoeffs'])[0][0][0]
                xim, yim = int(imgpts[0]), int(imgpts[1])
                if xim < 0 or xim > 1920 or yim < 0 or yim > 1080:
                    continue
                cv2.drawMarker(cam_img, (xim, yim), (255, 0, 0), markerType=cv2.MARKER_DIAMOND, markerSize=msize, thickness=thick, line_type=cv2.LINE_AA)
            
            # Save individual camera view
            save_im_path = os.path.join(json_root_path, im_name)
            cv2.imwrite(save_im_path, cam_img)

        # Add summary text to combined visualization
        cv2.putText(combined_frames, 'Ts: ' + str(sample['date']), (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined_frames, 'Dets: ' + str(len(pred_locs)), (1000, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined_frames, 'Gts: ' + str(len(gt_locs)), (1200, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Save combined visualization
        img_save_path = save_image_path + '/combined_per_sample/sample_' + str(sample['sample_number']) + '.jpg'
        os.makedirs(save_image_path + '/combined_per_sample', exist_ok=True)
        cv2.imwrite(img_save_path, combined_frames)
    
    return result_dict