import court_line_detector
from utils import (read_video, 
                   save_video)

from trackers import PlayerTracker, BallTracker

from court_line_detector import CourtLineDetector


def main():
    input_video_path = "input_videos/input_video.mp4"
    output_video_path = "output_videos/output_video.avi"

    # Read Vid
    video_frames = read_video(input_video_path)

    video_frames_copy = video_frames.copy()

    #Player Tracker
    player_tracker = PlayerTracker(model_path='models/yolo11x.pt')

    #Ball Tracker
    ball_tracker = BallTracker(model_path='models/tennis_ball_yolo11n_last.pt')

    #Detect Players
    player_detections = player_tracker.detect_frames(video_frames, 
                                                     read_from_stub=True,
                                                     stub_path='tracker_stubs/player_detections.pkl')
    
    #Detect Ball
    ball_detections = ball_tracker.detect_frames(video_frames, 
                                                 read_from_stub=False,
                                                 stub_path='tracker_stubs/ball_detections.pkl')
    
    # Court Line Detector
    court_model_path = "models/keypoints_model_last.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])


    #Draw Output
    ##Draw Player Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames_copy, player_detections)

    ##Draw Ball Boxes
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    #Draw Court Keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints) 

    # Save Output
    save_video(output_video_frames, output_video_path)

    return


if __name__ == "__main__":
    main()