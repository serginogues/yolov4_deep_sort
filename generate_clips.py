"""
Extract one clip per tracked object in a video
"""
import os
import cv2
import argparse
from pathlib import Path
import numpy as np
from deep_sort.tracker import Tracker
from yolo.yolov5utils import Detector
import anomaly_detection.preprocess as pre


def main(config):
    video_list = pre.get_video_paths(config.input)
    yolo_model = Detector()
    vid_count = 0
    for filename in video_list:
        process_video(filename, yolo_model)
        vid_count += 1


def process_video(filename: str, yolo_model, image_size: int):
    repo = os.path.splitext(filename)[0]
    create_empty_repo(repo)

    tracker = Tracker()
    vid = cv2.VideoCapture(filename)

    clip_list = {}  # list of lists of frames
    frame_count = 0

    stride = 1

    while vid.isOpened():
        ret, frame = vid.read()  # capture frame-by-frame
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            break

        bboxes = yolo_model.forward(frame)

        if bboxes.size > 0:
            tracker.predict(frame, bboxes)

        if frame_count % stride == 0:
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                _bbox = track.to_tlbr()
                _id = str(track.track_id)
                crop_frame = crop(frame, _bbox)
                h, w, _ = crop_frame.shape
                if w * h > 7000:
                    if _id in clip_list:
                        clip_list[_id].append(crop_frame)

                    else:
                        clip_list[_id] = [crop_frame]

        print(filename + ', frame: ' + str(frame_count))
        frame_count += 1

    cv2.destroyAllWindows()
    vid.release()

    # save clips as .npy
    repo = os.path.splitext(filename)[0]
    create_empty_repo(repo)

    for obj in clip_list.keys():
        l = clip_list[obj]
        if len(l) > 29:
            l = np.array(clip_list[obj], dtype=object)
            if len(l) > 125:
                l = l[15:115]
            np.save(os.path.join(repo, 'clip_' + obj), l)


def crop(img, bbox):
    return img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]


def create_empty_repo(file_path: str):
    try:
        Path(file_path).mkdir(parents=True, exist_ok=True)
        for f in os.listdir(file_path):
            os.remove(os.path.join(file_path, f))
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))


if __name__ == '__main__':
    # pre.show_clips(r'clips')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default=r'normal_clips',
                        help='path to input repo with videos to process')
    config = parser.parse_args()
    main(config)
