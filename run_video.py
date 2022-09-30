"""
In cmd shell:
>> python run_video.py
"""
import cv2
import argparse
import time
from deep_sort.tracker import Tracker
from yolov5utils import Detector


def run_video(config):
    """
    MAINSCRIPT
    """
    SAVE = config.save
    VIDEO_PATH = config.input

    # init
    yolo_model = Detector()
    tracker = Tracker()

    # begin video capture
    vid = cv2.VideoCapture(VIDEO_PATH)

    # get video ready to save locally if flag is set
    out = None
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if SAVE:
        # by default VideoCapture returns float instead of int
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*"mp4")  # 'XVID'
        out = cv2.VideoWriter("output", codec, fps, (frame_width, frame_height))

    # init display params
    start = time.time()
    counter = 0
    video_clip = []

    # read frame by frame until video is completed
    while vid.isOpened():

        ret, frame = vid.read()  # capture frame-by-frame
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_clip.append(frame)
        else:
            print('Video has ended or failed')
            break

        start_time = time.time()
        counter += 1
        print("Frame #", counter)

        # YOLO + Deep SORT
        bboxes = yolo_model.forward(frame)
        tracker.predict(frame, bboxes)

        # display tracks
        frame = tracker.display(frame)

        # checking video frame rate
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)

        # Writing FrameRate on video
        cv2.putText(frame, str(int(fps)) + " fps", (50, 70), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

        # convert back to BGR
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # if flag save video, else display
        if SAVE:
            out.write(result)
        else:
            # show frame
            cv2.imshow("Output Video", result)

            # Press Q on keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Closes all the frames
    cv2.destroyAllWindows()

    # Average fps
    end = time.time()
    seconds = end - start
    print("Time taken: {0} seconds".format(seconds))
    print("Number of frames: {0}".format(counter))
    fps = counter / seconds
    print("Estimated frames per second: {0}".format(fps))

    # When everything done, release the video capture object
    vid.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="test.mp4",
                        help='path to input video')
    parser.add_argument('--save', type=bool, default=False,
                        help='if TRUE, saves video in mp4 format, else shows live video')
    config = parser.parse_args()
    run_video(config)
