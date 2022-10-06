import os
import numpy as np
import cv2


def preprocess(clip: np.ndarray, clip_size: int, image_size: int):
    """
    Parameters
    ----------
    clip
        nested array of shape (clip_size,) (w, h, 3=BGR)
    """
    new_clip = []
    temporal_stride = max(int(len(clip) / clip_size), 1)
    for count in range(clip_size):
        img = clip[count * temporal_stride]
        #TODO: there was a bug here, train with RGB2GRAY
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (image_size, image_size)) / 256.0
        img = np.reshape(img, (image_size, image_size, 1))
        new_clip.append(img)
    return np.array(new_clip)


def show_clip(f):
    clip = np.load(f, allow_pickle=True)
    print(f)
    for im in clip:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        # im = np.resize(im, (192, 192, 3))
        cv2.imshow(f, im)
        cv2.waitKey(33)
    cv2.destroyAllWindows()


def get_video_paths(parent_path: str, _ext: list[str] = ['mp4', 'avi']) -> list[str]:
    l = []
    for root, dirs, files in os.walk(parent_path):
        for file in files:
            filename, extension = os.path.splitext(file)
            if any(word in extension.lower() for word in _ext):
                full_path = os.path.join(root, file)
                l.append(full_path)
    return l
