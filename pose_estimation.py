"""
How to use?
    >> import cv2
    >> img_origin = cv2.imread('test_videos/cropped18.jpg')
    >> img = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
    >> pose = new_instance()  # for each person to be detected
    >> result = pose.process(img)
    >> result_yolo = crop['im']

Mediapipe Output:
- x and y: Landmark coordinates normalized to [0.0, 1.0] by the image width and height respectively
- z: Represents the landmark depth with the depth at the midpoint of hips being the origin, and the smaller the value
the closer the landmark is to the camera. The magnitude of z uses roughly the same scale as x
- visibility: A value in [0.0, 1.0] indicating the likelihood of the landmark being visible (present and not occluded) in the image.
"""
import mediapipe as mp
import numpy as np

COMPLEXITY_POSE = 0  # 0, 1 or 2. The higher the more accurate, but slower

# mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def new_instance():
    return mp_pose.Pose(static_image_mode=True,
                        model_complexity=COMPLEXITY_POSE,
                        smooth_segmentation=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                        )


def pose_process_img(img: np.ndarray, pose) -> list:
    """
    :param img: image from a single person
    :param pose: mediapipe pose instance
    :return: keypoint landmarks
    """
    results = pose.process(img)
    keypoints = []
    if results.pose_landmarks:
        h, w, c = img.shape
        for lm in results.pose_landmarks.landmark:
            keypoints.append(landmarks_to_keypoints(w, h, lm))
    return results


def landmarks_to_keypoints(w, h, lm):
    """
    :param w: width of the image
    :param h: height of the image
    :param lm: results.pose_landmarks.landmark[i]
    :return: relative landmark coordinates to the image
    """
    cx, cy = int(lm.x * w), int(lm.y * h)
    return cx, cy


"""for count, kp in enumerate(frame_list_keypoints):
    img = new_frames[count]
    img = np.ascontiguousarray(img, dtype=np.uint8)
    for id, tuple_ in enumerate(kp):
        cx, cy = tuple_
        cv2.circle(img, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    # cv2.imwrite("frame_test%d.jpg" % count, img)"""