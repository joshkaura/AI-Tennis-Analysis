
from utils import (read_video, 
                   save_video)

from trackers import PlayerTracker, BallTracker

from court_line_detector import CourtLineDetector

import cv2

from minicourt import MiniCourt


def main():
    input_video_path = "input_videos/input_video.mp4"
    output_video_path = "output_videos/output_video.avi"

    homography = True

    # Read Vid
    video_frames = read_video(input_video_path)

    video_frames_copy = video_frames.copy()

    #Player Tracker
    player_tracker = PlayerTracker(model_path='models/yolo11x.pt')

    #Ball Tracker
    ball_tracker = BallTracker(model_path='models/tennis_ball_yolo11s_best.pt')

    #Detect Players
    player_detections = player_tracker.detect_frames(video_frames, 
                                                     read_from_stub=True,
                                                     stub_path='tracker_stubs/player_detections.pkl')
    
    #Detect Ball
    ball_detections = ball_tracker.detect_frames(video_frames, 
                                                 read_from_stub=True,
                                                 stub_path='tracker_stubs/ball_detections.pkl')
    
    # Interpolate 
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    #print(ball_detections)
    
    # Court Line Detector
    court_model_path = "models/keypoints_model_best.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Choose Players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    #Mini Court
    mini_court = MiniCourt(video_frames[0])

    #Detect Shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    print(f"{len(ball_shot_frames)} hits at frames: {ball_shot_frames}")

    if homography == False:
        # Convert positions to mini court positions
        player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, 
                                                                                                            ball_detections,
                                                                                                            court_keypoints)
        #print(ball_mini_court_detections)
    else:
        #*** Homography Version ***
        player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates_homography(
            player_detections, ball_detections, court_keypoints
        )

    #Draw Output
    ##Draw Player Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames_copy, player_detections)

    ##Draw Ball Boxes
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    #Draw Court Keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints) 

    #Draw Mini Court
    print("frames:", len(output_video_frames))
    print("player positions:", len(player_mini_court_detections))
    print("ball positions:", len(ball_mini_court_detections))

    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    #Draw Player Positions on Mini Court
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    #Draw Ball Position on Mini Court
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0,255,255))

    #print(ball_mini_court_detections)

    #Draw Frame Number (top left corner)
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Save Output
    save_video(output_video_frames, output_video_path)

    return


if __name__ == "__main__":
    main()