import numpy as np
import cv2

def convert_pixel_distance_to_meters(pixel_distance, reference_height_in_meters, reference_height_in_pixels):
    return (pixel_distance * reference_height_in_meters) / reference_height_in_pixels

def convert_meters_to_pixel_distance(meters, reference_height_in_meters, reference_height_in_pixels):
    return (meters * reference_height_in_pixels) / reference_height_in_meters

def _kp_list_to_points(keypoints, indices):
    """keypoints is [x0,y0,x1,y1,...]; indices are keypoint ids like [0,1,2,3]."""
    pts = []
    for i in indices:
        x = keypoints[i * 2]
        y = keypoints[i * 2 + 1]
        if x is None or y is None:
            return None
        pts.append([float(x), float(y)])
    return np.array(pts, dtype=np.float32)

def _warp_point(H, point):
    """point is (x,y) -> (x',y') using homography H."""
    pt = np.array([[point]], dtype=np.float32)  # (1,1,2)
    out = cv2.perspectiveTransform(pt, H)[0][0]
    return float(out[0]), float(out[1])