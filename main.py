from utils import (read_video, 
                   save_video)

from trackers import PlayerTracker


def main():
    input_video_path = "input_videos/input_video.mp4"
    output_video_path = "output_videos/output_video.avi"

    # Read Vid
    video_frames = read_video(input_video_path)

    #Detect Players
    player_tracker = PlayerTracker(model_path='models/yolo11x.pt')
    player_detections = player_tracker.detect_frames(video_frames, 
                                                     read_from_stub=True,
                                                     stub_path='tracker_stubs/player_detections.pkl')

    #Draw Output
    ##Draw Player Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)

    save_video(output_video_frames, output_video_path)

    return


if __name__ == "__main__":
    main()