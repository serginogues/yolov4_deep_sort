"""
In cmd shell:
>> python tracker.py
"""
# from PIL import Image
import argparse
import time
import numpy as np
from self_utils import *

# yolov5 imports
import torch
from matplotlib import pyplot as plt

# deep sort imports
# import utils
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet


def run_video(config):
    """
    MAINSCRIPT
    :param USE_CUDA: if True, sets yolo to 'cuda' if available, else 'cpu'
    :param TRACK_ONLY: classes id to track, other bboxes will be ignored
    :param SAVE: if TRUE, saves video in mp4 format, else shows live video
    :param VIDEO_PATH: path to input video
    """
    SAVE = config.save
    VIDEO_PATH = config.input
    YOLO_CONF = 0.5  # YOLO minimum confidence threshold
    DEEPSORT_WEIGHTS = "backup/mars-small128.pb"  # path to deep sort pre-trained cnn

    # init yolo
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo_model.to(device)
    print("YOLO device: ", device)
    yolo_model.classes = 0
    yolo_model.multi_label = False
    yolo_model.conf = YOLO_CONF

    # init deep sort
    # the encoder is a CNN pre-trained deep_SORT tracking model
    encoder = gdet.create_box_encoder(DEEPSORT_WEIGHTS, batch_size=1)

    # the Tracker takes care of creation, keeping track, and eventual deletion of all tracks
    # matching threshold = max cosine distance
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", matching_threshold=0.7)
    tracker = Tracker(metric)

    # begin video capture
    # if the input is the camera, pass 0 instead of the video path
    try:
        vid = cv2.VideoCapture(VIDEO_PATH)
    except:
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

    # initialize color map
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    # init video_clip for slowfast
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

        # YOLO
        """
        - results.pandas().xyxy[0]  # to obtain pd dict from single predicted image
        - results.xyxy[0] = results.pred[0]  # bboxes 
        - results.pandas().xywh[0]
        - results.xywh[0]
        - Crops:
                - crops[i]['im']  # cropped image
                - crops[i]['box']  # box coordinates
                - crops[i]['conf'].item()  # prediction confidence   where @i is [0, n crops)
        """

        results = yolo_model(frame)
        # crops = results.crop(save=False)
        bboxes = results.xyxy[0]

        # yolo to deep sort
        boxes, scores = xyxy_to_xywh(bboxes.cpu())

        # DEEP SORT
        """
        Deep Sort Detection requires boxes input in the following format (x_min, y_min, width, height), 
        but our YOLO outputs detections as (x_min, y_min, x_max, y_max)
        """
        # encode yolo detections and feed to tracker
        features = encoder(frame, boxes)

        # Detection input tlwh, confidence, feature
        detections = [Detection(bbox, score, feature) for bbox, score, feature in
                      zip(boxes, scores, features)]

        # Run non-maxima suppression.
        nms_max_overlap = 0.7
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker and update tracks
        tracker.predict()
        tracker.update(detections)

        # Slowfast
        # run it every 29 frames
        if len(video_clip) % 32 == 0:
            pass

        # Display
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()

            # draw person bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            # bbox
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            # info bbox
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                          (int(bbox[0]) + (len(str(track.track_id))) * 17, int(bbox[1])), color, -1)
            # text
            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                        (255, 255, 255), 2)
            # centroid
            cv2.circle(frame, center_from_bbox(bbox), 4, color, 2)
            # print details about each track
            print("Tracker ID: {}, BBox Coords (xmin, ymin, xmax, ymax): {}".format(
            str(track.track_id), (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # Display
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
