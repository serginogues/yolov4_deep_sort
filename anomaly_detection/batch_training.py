import tensorflow as tf
import numpy as np
from .preprocess import preprocess


class BatchTraining(tf.keras.utils.Sequence):
    def __init__(self, filenames, batch_size, clip_size, image_size):
        self.filenames = filenames
        self.batch_size = batch_size
        self.clip_size = clip_size
        self.image_size = image_size

    def __len__(self):
        return (np.ceil(len(self.filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_x = np.array([
                            preprocess(np.load(x, allow_pickle=True), self.clip_size, self.image_size)
                            for x in batch_x])
        return batch_x, batch_x