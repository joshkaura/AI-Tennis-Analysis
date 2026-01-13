import math
'''

def get_centre_bbox(bbox):
    x1, y1, x2, y2 = bbox
    centre_x = int((x1+x2)/2)
    centre_y = int((y1+y2)/2)
    return(centre_x, centre_y)

def measure_distance(p1,p2):
    distance = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
    
'''


def get_centre_bbox(bbox):
    #bbox: [x1,y1,x2,y2] -> (cx, cy) or None
    if bbox is None or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox

    # handle None values
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None

    # cast to float (works if they are ints, floats, numpy scalars, etc.)
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def measure_distance(p1, p2):
    #p1,p2: (x,y) -> float distance or None
    if p1 is None or p2 is None:
        return None
    x1, y1 = p1
    x2, y2 = p2
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None

    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    return math.hypot(x1 - x2, y1 - y2)

    
def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)

def get_closest_keypoint_index(point, keypoints, keypoint_indices):
   closest_distance = float('inf')
   key_point_ind = keypoint_indices[0]
   for keypoint_index in keypoint_indices:
       keypoint = keypoints[keypoint_index*2], keypoints[keypoint_index*2+1]
       distance = abs(point[1]-keypoint[1])

       if distance<closest_distance:
           closest_distance = distance
           key_point_ind = keypoint_index
    
   return key_point_ind

def get_height_of_bbox(bbox):
    return bbox[3]-bbox[1]

def measure_xy_distance(p1,p2):
    return abs(p1[0]-p2[0]), abs(p1[1]-p2[1])

def get_center_of_bbox(bbox):
    return (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))

def _ball_size(bbox):
    x1,y1,x2,y2 = bbox
    return max((x2-x1), (y2-y1))

def _y_net_from_drawing_kps(dkps):
    y_top = dkps[1]          # point 0 y
    y_bottom = dkps[5]       # point 2 y (since point 2 is at [4],[5] in your mini court)
    return (y_top + y_bottom) / 2.0

def depth_adjust_end_on(mini_xy, y_net, y_top, y_bottom, k_far=50.0, k_near=25.0, gamma=2.0):
    """
    Adjust ball depth based on how far from the net it is.
    - far side: pull toward net (reduce y) more as it approaches far baseline
    - near side: push away from net (increase y) more as it approaches near baseline

    y_top    = top baseline y (mini court)
    y_bottom = bottom baseline y (mini court)
    """
    x, y = mini_xy

    # far side (top half) if your mini-court y increases downward:
    if y < y_net:
        # t: 0 at net, 1 at top baseline
        denom = max(1.0, (y_net - y_top))
        t = (y_net - y) / denom
        t = max(0.0, min(1.0, t))
        shift = k_far * (t ** gamma)
        y = y + shift  # if y<y_net, adding moves it DOWN toward net? depends on axis (see note below)
    else:
        # near side (bottom half)
        denom = max(1.0, (y_bottom - y_net))
        t = (y - y_net) / denom
        t = max(0.0, min(1.0, t))
        shift = k_near * (t ** gamma)
        y = y + shift

    return (x, y)