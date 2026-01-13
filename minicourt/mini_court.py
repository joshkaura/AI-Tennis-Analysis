import cv2
import sys
import numpy as np

sys.path.append('../')
import constants
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    measure_distance,
    get_centre_bbox,
    _warp_point,
    _kp_list_to_points,
    depth_adjust_end_on,
    _y_net_from_drawing_kps,
    _ball_size
)

class MiniCourt():
    def __init__(self,frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court=20

        # NEW: extra padding around the court in meters (real-world)
        #self.extra_padding_meters = 2.0  # tweak (e.g., 1.5–3.0)

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()


    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(meters,
                                                constants.DOUBLE_LINE_WIDTH,
                                                self.court_drawing_width
                                            )

    def set_court_drawing_key_points(self):
        drawing_key_points = [0]*28

        # point 0 
        drawing_key_points[0] , drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1
        drawing_key_points[2] , drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # point 2
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT*2)
        # point 3
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5] 
        # #point 4
        drawing_key_points[8] = drawing_key_points[0] +  self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1] 
        # #point 5
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5] 
        # #point 6
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3] 
        # #point 7
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7] 
        # #point 8
        drawing_key_points[16] = drawing_key_points[8] 
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 9
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17] 
        # #point 10
        drawing_key_points[20] = drawing_key_points[10] 
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 11
        drawing_key_points[22] = drawing_key_points[20] +  self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21] 
        # # #point 12
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18])/2)
        drawing_key_points[25] = drawing_key_points[17] 
        # # #point 13
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22])/2)
        drawing_key_points[27] = drawing_key_points[21] 

        self.drawing_key_points=drawing_key_points

    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6,7),
            (1,3),
            
            (0,1),
            (8,9),
            (10,11),
            (10,11),
            (2,3)
        ]

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def set_canvas_background_box_position(self,frame):
        frame= frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def draw_court(self,frame):
        for i in range(0, len(self.drawing_key_points),2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i+1])
            cv2.circle(frame, (x,y),5, (0,0,255),-1)

        # draw Lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Draw net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame

    def draw_background_rectangle(self,frame):
        shapes = np.zeros_like(frame,np.uint8)
        # Draw the rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha=0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return out

    def draw_mini_court(self,frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        print("Drawn Mini Court")
        return output_frames

    def get_start_point_of_mini_court(self):
        return (self.court_start_x,self.court_start_y)
    def get_width_of_mini_court(self):
        return self.court_drawing_width
    def get_court_drawing_keypoints(self):
        return self.drawing_key_points

    def get_mini_court_coordinates(self,
                                   object_position,
                                   closest_key_point, 
                                   closest_key_point_index, 
                                   player_height_in_pixels,
                                   player_height_in_meters
                                   ):
        
        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(object_position, closest_key_point)

        # Conver pixel distance to meters
        distance_from_keypoint_x_meters = convert_pixel_distance_to_meters(distance_from_keypoint_x_pixels,
                                                                           player_height_in_meters,
                                                                           player_height_in_pixels
                                                                           )
        distance_from_keypoint_y_meters = convert_pixel_distance_to_meters(distance_from_keypoint_y_pixels,
                                                                                player_height_in_meters,
                                                                                player_height_in_pixels
                                                                          )
        
        # Convert to mini court coordinates
        mini_court_x_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_x_meters)
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_y_meters)
        closest_mini_coourt_keypoint = ( self.drawing_key_points[closest_key_point_index*2],
                                        self.drawing_key_points[closest_key_point_index*2+1]
                                        )
        
        mini_court_player_position = (closest_mini_coourt_keypoint[0]+mini_court_x_distance_pixels,
                                      closest_mini_coourt_keypoint[1]+mini_court_y_distance_pixels
                                        )

        return  mini_court_player_position

    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, original_court_key_points):
        player_heights = {
            1: constants.PLAYER_1_HEIGHT_METERS,
            2: constants.PLAYER_2_HEIGHT_METERS
        }

        output_player_boxes = []
        output_ball_boxes = []

        for frame_num, player_bbox in enumerate(player_boxes):

            # ALWAYS create outputs for this frame
            output_player_bboxes_dict = {}
            ball_out = {}  # default: no ball this frame

            # --- ball may be missing ---
            ball_position = None
            if frame_num < len(ball_boxes):
                ball_box = ball_boxes[frame_num][1]
                ball_position = get_centre_bbox(ball_box)

            # If no players detected, still append empties
            if not player_bbox:
                output_player_boxes.append(output_player_bboxes_dict)
                output_ball_boxes.append(ball_out)
                continue

            # Find closest player to ball (only if ball_position exists)
            closest_player_id_to_ball = None
            if ball_position is not None:
                valid_distances = []
                for player_id, bbox in player_bbox.items():
                    player_center = get_centre_bbox(bbox)
                    if player_center is None:
                        continue
                    dist = measure_distance(ball_position, player_center)
                    if dist is None:
                        continue
                    valid_distances.append((player_id, dist))

                if valid_distances:
                    closest_player_id_to_ball = min(valid_distances, key=lambda x: x[1])[0]

            # Compute mini-court player positions
            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)

                closest_key_point_index = get_closest_keypoint_index(
                    foot_position, original_court_key_points, [0, 2, 12, 13]
                )
                closest_key_point = (
                    original_court_key_points[closest_key_point_index * 2],
                    original_court_key_points[closest_key_point_index * 2 + 1],
                )

                frame_index_min = max(0, frame_num - 20)
                frame_index_max = min(len(player_boxes), frame_num + 50)

                # guard: player might not exist in all frames
                heights = []
                for i in range(frame_index_min, frame_index_max):
                    if player_id in player_boxes[i]:
                        heights.append(get_height_of_bbox(player_boxes[i][player_id]))
                if not heights:
                    continue

                max_player_height_in_pixels = max(heights)

                mini_pos = self.get_mini_court_coordinates(
                    foot_position,
                    closest_key_point,
                    closest_key_point_index,
                    max_player_height_in_pixels,
                    player_heights.get(player_id, constants.PLAYER_1_HEIGHT_METERS)
                )

                output_player_bboxes_dict[player_id] = mini_pos

                #print(ball_position)
                # Compute mini-court ball position using the closest player scale
                if closest_player_id_to_ball == player_id and ball_position is not None:
                    ckpi = get_closest_keypoint_index(ball_position, original_court_key_points, [0, 2, 12, 13])
                    ckpt = (
                        original_court_key_points[ckpi * 2],
                        original_court_key_points[ckpi * 2 + 1]
                    )
                    #print(ball_position)
                    ball_mini_pos = self.get_mini_court_coordinates(
                        ball_position,
                        ckpt,
                        ckpi,
                        max_player_height_in_pixels,
                        player_heights.get(player_id, constants.PLAYER_1_HEIGHT_METERS)
                    )

                    ball_out = {1: ball_mini_pos}
                    #print(ball_out)

            # ALWAYS append per frame
            output_player_boxes.append(output_player_bboxes_dict)
            output_ball_boxes.append(ball_out)

            if frame_num < 5:
                print("frame", frame_num,
                    "ball_position None?", ball_position is None,
                    "closest_player:", closest_player_id_to_ball,
                    "valid_distances_len:", 0 if ball_position is None else len(valid_distances))

        return output_player_boxes, output_ball_boxes
    
    # *** Homography Based Version ***
    def compute_court_to_mini_homography(self, original_court_key_points, indices):
        self.src = _kp_list_to_points(original_court_key_points, indices)  # image
        self.dst = _kp_list_to_points(self.drawing_key_points, indices)      # mini court

        if self.src is None or self.dst is None:
            return None

        # Use RANSAC for robustness if keypoints are a bit noisy
        H, _ = cv2.findHomography(self.src, self.dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        return H
    
    def convert_bounding_boxes_to_mini_court_coordinates_homography(
        self,
        player_boxes,
        ball_boxes,
        original_court_key_points,
        indices=(0, 1, 2, 3),
        k_far=90.0,
        k_near=75.0,
        gamma=2.0
    ):
        """
        Returns:
          output_player_positions: list[dict[player_id -> (x,y)]]
          output_ball_positions:   list[dict[1 -> (x,y)]]
        """

        output_player_positions = []
        output_ball_positions = []

        # 1) Homography
        H = self.compute_court_to_mini_homography(original_court_key_points, indices)

        # Fail-safe: return aligned empty lists
        if H is None:
            for _ in range(len(player_boxes)):
                output_player_positions.append({})
                output_ball_positions.append({})
            return output_player_positions, output_ball_positions

        # 2) Mini-court reference ys
        # top baseline y = point 0 y (drawing_key_points[1])
        # bottom baseline y = point 2 y (drawing_key_points[5])
        y_top = float(self.drawing_key_points[1])
        y_bottom = float(self.drawing_key_points[5])
        y_net = (y_top + y_bottom) / 2.0

        n_frames = len(player_boxes)

        for frame_num in range(n_frames):
            player_bbox_dict = player_boxes[frame_num] or {}
            ball_bbox_dict = ball_boxes[frame_num] if frame_num < len(ball_boxes) else {}

            # --- players ---
            players_out = {}
            for player_id, bbox in player_bbox_dict.items():
                foot = get_foot_position(bbox)            # (x,y) in image pixels
                players_out[player_id] = _warp_point(H, foot)

            # --- ball ---
            ball_out = {}
            if isinstance(ball_bbox_dict, dict) and 1 in ball_bbox_dict:
                ball_bbox = ball_bbox_dict[1]
                ball_center = get_centre_bbox(ball_bbox)  # (x,y) in image pixels
                if ball_center is not None:
                    mini_ball = _warp_point(H, ball_center)

                    # 3) Depth adjustment (stronger further from net)
                    mini_ball = depth_adjust_end_on(
                        mini_ball, y_net=y_net, y_top=y_top, y_bottom=y_bottom,
                        k_far=k_far, k_near=k_near, gamma=gamma
                    )

                    ball_out = {1: mini_ball}

            output_player_positions.append(players_out)
            output_ball_positions.append(ball_out)

        return output_player_positions, output_ball_positions

    
    def draw_points_on_mini_court(self,frames,positions, color=(0,255,0)):
        for frame_num, frame in enumerate(frames):
            for _, position in positions[frame_num].items():
                x,y = position
                x= int(x)
                y= int(y)
                cv2.circle(frame, (x,y), 5, color, -1)
        return frames