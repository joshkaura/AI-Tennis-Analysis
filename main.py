from utils import (read_video, 
                   save_video)



def main():
    input_video_path = "input_videos/input_video.mp4"
    output_video_path = "output_videos/output_video.avi"

    video_frames = read_video(input_video_path)

    save_video(video_frames, output_video_path)

    return


if __name__ == "__main__":
    main()