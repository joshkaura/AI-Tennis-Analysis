

def get_centre_bbox(bbox):
    x1, y1, x2, y2 = bbox
    centre_x = int((x1+x2)/2)
    centre_y = int((y1+y2)/2)
    return(centre_x, centre_y)

def measure_distance(p1,p2):
    distance = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
    