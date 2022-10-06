import os
import cv2


def path_to_video_frames_list(path: str = 'test_videos/video-dancing.mp4', SAVE: bool = False):
    """
    Convert a video into a frame list
    :param SAVE: if True, saves all frames
    :param path: path to video
    :return: list of frames (cv2 images)
    """
    frames = []
    vidcap = cv2.VideoCapture(path)  # get video
    success, image = vidcap.read()  # get first frame
    frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    img_name = path.split(".")[0]
    count = 0
    while success:
        success, image = vidcap.read()  # get current frame
        if image is not None:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(img)
            if SAVE:
                cv2.imwrite(img_name + "_" + str(count) + ".jpg", image)
            count += 1
    return frames


def read_classes(path: str) -> list:
    """
    Read darknet labels file
    :param path: path to classes.txt
    :return: list of classes names
    """
    names = []
    with open(path, 'r') as data:
        for name in data:
            names.append(name.strip('\n'))
    return names


def get_labels_from_txt(path) -> list:
    """
    :param path: path to labels repository
    :return: bboxes
    """
    bboxes = [None] * len([name for name in os.listdir('data/videos_tornos/Canal_018_FraudeTornos/') if name.endswith(".txt")])

    for file in os.listdir(path):
        if file.endswith(".txt"):
            idx = int(file.split(".")[0].split("_")[-1])
            bboxes[idx] = read_labels(file)
    return bboxes


def read_labels(path: str, LABELS: list = [3, 6]) -> list:
    """
    Reads label file and stores only LABELS in [x, y, width, height, label_id] format \n
    :param LABELS: list of labels to be considered, the rest is discarded
    :param path: path to YOLO labels txt file
    :return: list of [x, y, width, height, label_id]
    """
    object_list = []
    with open(TURNSTILES_PATH + path, 'r') as data:
        for line in data:
            x = line.split(" ")
            if int(x[0]) in LABELS:
                object_list.append([float(x[1]), float(x[2]), float(x[3]), float(x[4]), int(x[0])])
    return object_list


def labelImg_to_yolo_format(bboxes, image_width, image_height):
    """
    [x, y, width, height, label_id] to [xmin, ymin, xmax, ymax, label_id] \n
    :param bboxes: frame detections
    :param image_height: height of the frame
    :param image_width: width of the frame
    """
    new_boxes = []
    for box in bboxes:
        xcenter = int(box[0] * image_width)
        ycenter = int(box[1] * image_height)
        width = int(box[2] * image_width)
        height = int(box[3] * image_height)

        xmin = xcenter - (width/2)
        ymin = ycenter - (height/2)
        xmax = xcenter + (width/2)
        ymax = ycenter + (height/2)

        new_boxes.append([xmin, ymin, xmax, ymax, box[4]])
    return new_boxes


def center_from_bbox(bbox):
    """
    :param bbox: [xmin, ymin, xmax, ymax, label_id]
    :return: (x,y)
    """
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[2]
    ymax = bbox[3]
    return int(xmin + (xmax - xmin) / 2), int(ymin + (ymax - ymin) / 2)


def is_above(bbox1, bbox2) -> bool:
    """
    :param y1: y coordinates of object 1
    :param y2: y coordinates of object 2
    :return: True if 1 above 2
    """
    _, y1 = center_from_bbox(bbox1)
    _, y2 = center_from_bbox(bbox2)
    return True if y1 < y2 else False


# path_to_video_frames_list("data/videos_tornos/Canal_018_FraudeTornos_img/Canal_018_FraudeTornos.mp4", SAVE=True)