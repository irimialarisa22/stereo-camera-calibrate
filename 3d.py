import cv2
import numpy as np


def measure_dist(stereo: {},
                 a_l: (float, float),
                 b_l: (float, float),
                 a_r: (float, float),
                 b_r: (float, float)) -> float:
    lxy1 = np.array(a_l, dtype=float)
    lxy1 = cv2.undistortPoints(src=lxy1,
                               cameraMatrix=stereo['M1'],
                               distCoeffs=stereo['D1'],
                               R=stereo['R1'],
                               P=stereo['P1']).flatten()
    rxy1 = np.array(a_r, dtype=float)
    rxy1 = cv2.undistortPoints(src=rxy1,
                               cameraMatrix=stereo['M2'],
                               distCoeffs=stereo['D2'],
                               R=stereo['R2'],
                               P=stereo['P2']).flatten()
    xy1 = cv2.triangulatePoints(projMatr1=stereo['P1'],
                                projMatr2=stereo['P2'],
                                projPoints1=lxy1.astype(dtype=float),
                                projPoints2=rxy1.astype(dtype=float))
    xy1 = cv2.convertPointsFromHomogeneous(xy1.T).flatten()

    lxy2 = np.array(b_l, dtype=float)
    lxy2 = cv2.undistortPoints(src=lxy2,
                               cameraMatrix=stereo['M1'],
                               distCoeffs=stereo['D1'],
                               R=stereo['R1'],
                               P=stereo['P1']).flatten()
    rxy2 = np.array(b_r, dtype=float)
    rxy2 = cv2.undistortPoints(src=rxy2,
                               cameraMatrix=stereo['M2'],
                               distCoeffs=stereo['D2'],
                               R=stereo['R2'],
                               P=stereo['P2']).flatten()
    xy2 = cv2.triangulatePoints(projMatr1=stereo['P1'],
                                projMatr2=stereo['P2'],
                                projPoints1=lxy2.astype(dtype=float),
                                projPoints2=rxy2.astype(dtype=float))
    xy2 = cv2.convertPointsFromHomogeneous(xy2.T).flatten()

    return np.linalg.norm(xy1 - xy2)