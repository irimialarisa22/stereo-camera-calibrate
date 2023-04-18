import cv2


class Constants:
    CRIT_STEREO = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ROWS = 7
    COLUMNS = 10
    WORLD_SCALING = 9.937
    S_WIDTH = 1920
    S_HEIGHT = 1080