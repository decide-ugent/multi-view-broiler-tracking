import numpy as np
import cv2

def load_calibration(cam_number, calibration_root='../calibrations_BAA'):
    cameraMatrix = np.loadtxt(calibration_root + '/cam_' +
                              str(cam_number) + '/intrinsics/cameraMatrix.txt')
    distCoeffs = np.loadtxt(calibration_root + '/cam_' +
                            str(cam_number) + '/intrinsics/distCoeffs.txt')

    tvec = np.loadtxt(calibration_root + '/cam_' +
                      str(cam_number) + '/extrinsics/tvec.txt')
    rvec = np.loadtxt(calibration_root + '/cam_' +
                      str(cam_number) + '/extrinsics/rvec.txt')

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    translation_matrix = tvec.reshape(3, 1)
    extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))

    worldcoord2imgcoord_mat = cameraMatrix @ np.delete(extrinsic_matrix, 2, 1)
    imgcoord2worldcoord_mat = np.linalg.inv(worldcoord2imgcoord_mat)

    cam_location = ((-rotation_matrix.T).dot(translation_matrix)).reshape(3)

    return {'cameraMatrix': cameraMatrix,
            'distCoeffs': distCoeffs,
            'extrinsic_matrix': extrinsic_matrix,
            'rotation_matrix': rotation_matrix,
            'translation_matrix': translation_matrix,
            'worldcoord2imgcoord_mat': worldcoord2imgcoord_mat,
            'imgcoord2worldcoord_mat': imgcoord2worldcoord_mat,
            'tvec': tvec,
            'rvec': rvec,
            'cx': cameraMatrix[0, 2],
            'cy': cameraMatrix[1, 2],
            'fx': cameraMatrix[0, 0],
            'fy': cameraMatrix[1, 1],
            'cam_location': cam_location
            }

def project_2d_points_to_ground_plane(points, scale, imgcoord2worldcoord_mat):
    points = np.array(points, dtype=np.float64).reshape(-1, 2)
    points = np.hstack([points, np.ones((points.shape[0], 1))])
    X, Y, Z = imgcoord2worldcoord_mat @ points.T
    return np.array([(X/Z)*scale, (Y/Z)*scale]).T

def distance_of_2d_pixel_to_camera_new(calib, point, scalefactor=1000):
    pt_reshaped = np.array(point, dtype=np.float64).reshape(-1, 1, 2)
    undistorted_pt_reshaped = cv2.undistortPoints(pt_reshaped, calib['cameraMatrix'], calib['distCoeffs'], P=calib['cameraMatrix'])
    projected_point = project_2d_points_to_ground_plane(undistorted_pt_reshaped, scale=scalefactor, imgcoord2worldcoord_mat=calib['imgcoord2worldcoord_mat'])
    cam_loc = calib['cam_location'][:2] * scalefactor
    distance = np.linalg.norm(projected_point - cam_loc)
    return distance, cam_loc, projected_point

def project_bb_to_ground_plane(xmin, ymin, xmax, ymax, calib, min_dist, max_dist):
    centerpoint = [int((xmin + xmax)/2), int((ymin + ymax)/2)]
    # center_dist, _, _ = distance_of_2d_pixel_to_camera_new(calib, centerpoint, scalefactor=100)
    # factor = (center_dist - (min_dist/10)) / ((max_dist/10) - (min_dist/10))
    center_dist, cam_loc, _ = distance_of_2d_pixel_to_camera_new(calib, centerpoint, scalefactor=1000)
    factor = (center_dist - (min_dist)) / ((max_dist) - (min_dist))
    factor = 1 - np.clip(factor, 0, 1)
    box_selection = [int((xmin + xmax)/2), int(ymax - ((ymax - ymin)/2) * factor)]

    points_2d = np.array([box_selection], dtype=np.float32)
    undistorted_points_2d = cv2.undistortPoints(points_2d, calib['cameraMatrix'], calib['distCoeffs'], P=calib['cameraMatrix'])
    projected_points = project_2d_points_to_ground_plane(undistorted_points_2d, 100, calib['imgcoord2worldcoord_mat'])
    center_dist = np.linalg.norm(projected_points[0] - cam_loc/10)
    return projected_points, center_dist

def project_2d_points_to_3d_with_distortion(points, cameraMatrix, extrinsic_matrix, scale, distortion_coeffs):

    # print(points.shape)
    # print(points)
    # points  = points.reshape(-1, 1, 2)
    points = cv2.undistortPoints(points, cameraMatrix, distortion_coeffs, P=cameraMatrix)
    # print(points.shape) # (1, 1, 2)
    # points = [points[0][0][0], points[0][0][1], 1]
    # add a 1 to the end of each point
    points = np.array(points, dtype=np.float64).reshape(-1, 2)
    points = np.hstack([points, np.ones((points.shape[0], 1))])
    # print(points.shape)

    # reshape to (3, 1, 2)
    # points = points.reshape(-1, 1, 2)

    # print(points.shape)
    # print(points.T.shape)

    worldcoord2imgcoord_mat = cameraMatrix @ np.delete(extrinsic_matrix, 2, 1)
    proj_matrix = np.linalg.inv(worldcoord2imgcoord_mat)
    # print(proj_matrix.shape)
    X, Y, Z = proj_matrix @ points.T
    X = X/Z
    Y = Y/Z
    return np.array([X*scale, Y*scale]).T

def distance_of_2d_pixel_to_camera(calibs, cam_num, point, scalefactor=1000):
    # here we will use the calibration parameters to determine the distance of a pixel to the camera
    # print(point)

    x = point[0]
    y = point[1]
    # np. 
    point = np.array([x, y]).T.reshape(-1, 1, 2)
    # print(point)
    # to float 64
    point = np.array(point, dtype=np.float64)
    projected_point = project_2d_points_to_3d_with_distortion(point, 
                cameraMatrix=calibs[cam_num]['cameraMatrix'], 
                extrinsic_matrix=calibs[cam_num]['extrinsic_matrix'], 
                scale=scalefactor,
                distortion_coeffs=calibs[cam_num]['distCoeffs'], 
          )
    
    R = calibs[cam_num]['rotation_matrix']
    T = calibs[cam_num]['translation_matrix']
    world = (-R.T).dot(T)
    world = world.reshape(3)

    cam_loc = world[:2] * scalefactor
    distance = np.linalg.norm(projected_point - cam_loc)

    return distance, cam_loc, projected_point