import cv2
import numpy as np
from constants import Constants


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


if __name__ == '__main__':
    fs = cv2.FileStorage('sol.yml', cv2.FILE_STORAGE_READ)

    sol = {'M1': fs.getNode('M1').mat(),
           'M2': fs.getNode('M2').mat(),
           'D1': fs.getNode('D1').mat(),
           'D2': fs.getNode('D2').mat(),
           'R1': fs.getNode('R1').mat(),
           'R2': fs.getNode('R2').mat(),
           'P1': fs.getNode('P1').mat(),
           'P2': fs.getNode('P2').mat(),
           'R': fs.getNode('R').mat(),
           'T': fs.getNode('T').mat(),
           'E': fs.getNode('E').mat(),
           'F': fs.getNode('F').mat(),
           'Q': fs.getNode('Q').mat(),
           'SZ': (int(Constants.S_WIDTH), int(Constants.S_HEIGHT))}
    fs.release()


    dist = measure_dist(sol, (1239.6902, 434.91632), (649.37585, 833.39636), (1675.2928, 175.65314), (997.4407, 634.50916))
    print(dist)