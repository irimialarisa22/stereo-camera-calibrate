import cv2
import glob
import numpy as np
from constants import Constants


def detect_checkerboard_corners(image_path, criteria=Constants.CRIT_STEREO, rows=Constants.ROWS, columns=Constants.COLUMNS, world_scaling=Constants.WORLD_SCALING):
    # load the image and convert to grayscale
    frame = cv2.imread(image_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(image=gray,
                                             patternSize=(rows, columns),
                                             flags=None)

    if ret:
        # refine the corner positions
        corners = cv2.cornerSubPix(image=gray,
                                   corners=corners,
                                   winSize=(11, 11),
                                   zeroZone=(-1, -1),
                                   criteria=criteria)

        # draw the corners on the image
        cv2.drawChessboardCorners(image=frame,
                                  patternSize=(rows, columns),
                                  corners=corners,
                                  patternWasFound=ret)

        # scale the object points to match the checkerboard size
        objp = np.zeros((rows * columns, 3), np.float32)
        objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
        objp *= world_scaling

        return True, objp, corners, frame

    return False, None, None, None


def calibrate_cameras_for_intrinsic_parameters(image_paths_left, image_paths_right):

    # pixel coordinates of checkerboards
    img_points_l = []
    img_points_r = []

    # coordinates of the checkerboard in checkerboard world space
    obj_points = []

    for image_path_left, image_path_right in zip(image_paths_left, image_paths_right):
        # detect the checkerboard corners and get the object and image points
        ret_left, objp_left, corners_left, frame_left = detect_checkerboard_corners(image_path=image_path_left,
                                                                                    criteria=Constants.CRIT_STEREO,
                                                                                    rows=Constants.ROWS,
                                                                                    columns=Constants.COLUMNS,
                                                                                    world_scaling=Constants.WORLD_SCALING)
        ret_right, objp_right, corners_right, frame_right = detect_checkerboard_corners(image_path=image_path_right,
                                                                                        criteria=Constants.CRIT_STEREO,
                                                                                        rows=Constants.ROWS,
                                                                                        columns=Constants.COLUMNS,
                                                                                        world_scaling=Constants.WORLD_SCALING)

        if ret_left and ret_right:
            obj_points.append(objp_left)
            img_points_l.append(corners_left)
            img_points_r.append(corners_right)

            # display the images side by side
            frame_left = cv2.putText(img=frame_left,
                                     text='LEFT IMAGE',
                                     org=(20, 30),
                                     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                     fontScale=1,
                                     color=(255, 255, 255),
                                     thickness=2,
                                     lineType=cv2.LINE_AA)
            frame_right = cv2.putText(img=frame_right,
                                     text='RIGHT IMAGE',
                                     org=(20, 30),
                                     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                     fontScale=1,
                                     color=(255, 255, 255),
                                     thickness=2,
                                     lineType=cv2.LINE_AA)
            cv2.imshow('CALIBRATION IMAGES', np.hstack((frame_left, frame_right)))
            cv2.waitKey(500)

            key = cv2.waitKey(0)
            if key == ord('q'):
                break
            elif key == ord('n'):
                continue

    cv2.destroyAllWindows()
    ret_l, mtx_l, dst_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objectPoints=obj_points,
                                                                imagePoints=img_points_l,
                                                                imageSize=(Constants.S_WIDTH, Constants.S_HEIGHT),
                                                                cameraMatrix=None,
                                                                distCoeffs=None)
    ret_r, mtx_r, dst_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objectPoints=obj_points,
                                                                imagePoints=img_points_r,
                                                                imageSize=(Constants.S_WIDTH, Constants.S_HEIGHT),
                                                                cameraMatrix=None,
                                                                distCoeffs=None)

    print('LEFT CAMERA RMSE:', ret_l)
    print('LEFT CAMERA MATRIX:\n', mtx_l)
    print('LEFT CAMERA DISTORTION COEFFICIENTS:', dst_l)
    print('LEFT CAMERA ROTATION VECTORS:\n', rvecs_l)
    print('LEFT CAMERA TRANSLATION VECTORS:\n', tvecs_l)

    print('RIGHT CAMERA RMSE:', ret_r)
    print('RIGHT CAMERA MATRIX:\n', mtx_r)
    print('RIGHT CAMERA DISTORTION COEFFICIENTS:', dst_r)
    print('RIGHT CAMERA ROTATION VECTORS:\n', rvecs_r)
    print('RIGHT CAMERA TRANSLATION VECTORS:\n', tvecs_r)

    return mtx_l, dst_l, mtx_r, dst_r


def stereo_calibrate(mtx_l, dst_l, mtx_r, dst_r, image_paths_left, image_paths_right):
    img_points_l = []
    img_points_r = []

    obj_points = []

    for path_left, path_right in zip(image_paths_left, image_paths_right):
        # detect the checkerboard corners and get the object and image points
        ret_left, objp_left, corners_left, frame_left = detect_checkerboard_corners(image_path=path_left,
                                                                                    criteria=Constants.CRIT_STEREO,
                                                                                    rows=Constants.ROWS,
                                                                                    columns=Constants.COLUMNS,
                                                                                    world_scaling=Constants.WORLD_SCALING)
        ret_right, objp_right, corners_right, frame_right = detect_checkerboard_corners(image_path=path_right,
                                                                                        criteria=Constants.CRIT_STEREO,
                                                                                        rows=Constants.ROWS,
                                                                                        columns=Constants.COLUMNS,
                                                                                        world_scaling=Constants.WORLD_SCALING)
        if ret_left and ret_right:
            obj_points.append(objp_left)
            img_points_l.append(corners_left)
            img_points_r.append(corners_right)

    stereo_calibration_flags = cv2.CALIB_FIX_INTRINSIC
    err_rms, mtx_l, dst_l, mtx_r, dst_r, r, t, e, f = cv2.stereoCalibrate(objectPoints=obj_points,
                                                                          imagePoints1=img_points_l,
                                                                          imagePoints2=img_points_r,
                                                                          cameraMatrix1=mtx_l,
                                                                          distCoeffs1=dst_l,
                                                                          cameraMatrix2=mtx_r,
                                                                          distCoeffs2=dst_r,
                                                                          imageSize=(Constants.S_WIDTH, Constants.S_HEIGHT),
                                                                          criteria=Constants.CRIT_STEREO,
                                                                          flags=stereo_calibration_flags)

    r1, r2, p1, p2, q, roi_l, roi_r = cv2.stereoRectify(cameraMatrix1=mtx_l,
                                                        distCoeffs1=dst_l,
                                                        cameraMatrix2=mtx_r,
                                                        distCoeffs2=dst_r,
                                                        imageSize=(Constants.S_WIDTH, Constants.S_HEIGHT),
                                                        R=r,
                                                        T=t,
                                                        alpha=1,
                                                        flags=0)

    print('RMSE:', err_rms)
    print('T:', t)
    print('R1:', r1)
    print('R2:', r2)
    print('P1:', p1)
    print('P2:', p1)
    return r, t, r1, r2, p1, p2


if __name__ == '__main__':
    image_paths_left = glob.glob("img_for_calib/left/*.jpg")
    image_path_right = glob.glob("img_for_calib/right/*.jpg")

    mtx_l, dst_l, mtx_r, dst_r = calibrate_cameras_for_intrinsic_parameters(image_paths_left, image_path_right)
    r, t, r1, r2, p1, p2 = stereo_calibrate(mtx_l, dst_l, mtx_r, dst_r, image_paths_left, image_path_right)
