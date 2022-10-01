"""
Extract one clip per tracked object in a video
"""
import os
import cv2
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from deep_sort.tracker import Tracker
from yolov5utils import Detector


def main(config):
    video_list = get_video_paths(config.input)
    image_size = config.img_size
    yolo_model = Detector()
    vid_count = 0
    for filename in tqdm(video_list, desc='Generating clips from ' + video_list[vid_count]):
        process_video(filename, yolo_model, image_size)
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

        frame_count += 1

    cv2.destroyAllWindows()
    vid.release()

    for obj in clip_list.keys():
        # save clips as .npy
        l = np.array(clip_list[obj], dtype=object)
        if len(l) > 29:
            if len(l) > 125:
                l = l[15:115]
            np.save(os.path.join(repo, 'clip_' + obj), l)


def show_clips(dir):
    for f in os.listdir(dir):
        p = os.path.join(dir, f)
        if p.endswith('npy'):
            arr = np.load(p)
            show_clip(arr)


def show_clip(clip):
    for im in clip:
        cv2.imshow('', im)
        cv2.waitKey(0)


def crop(img, bbox):
    return img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]


def create_empty_repo(file_path: str):
    try:
        Path(file_path).mkdir(parents=True, exist_ok=True)
        for f in os.listdir(file_path):
            os.remove(os.path.join(file_path, f))
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))


def get_video_paths(parent_path: str, _ext: list[str] = ['mp4', 'avi']) -> list[str]:
    l = []
    for root, dirs, files in os.walk(parent_path):
        for file in files:
            filename, extension = os.path.splitext(file)
            if any(word in extension.lower() for word in _ext):
                full_path = os.path.join(root, file)
                l.append(full_path)
    return l


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=r'vids',
                        help='path to input repo with videos to process')
    parser.add_argument('--img_size', type=int, default=160,
                        help='resize')
    config = parser.parse_args()
    main(config)
