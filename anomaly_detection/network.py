import os
import cv2
import numpy as np
import tensorflow as tf
from keras.layers import Conv2DTranspose, ConvLSTM2D, TimeDistributed, Conv2D, LayerNormalization
from keras.models import Sequential, load_model
from .batch_training import BatchTraining
from .preprocess import preprocess, get_video_paths

seed_constant = 28
tf.random.set_seed(seed_constant)
np.random.seed(seed=seed_constant)


class AnomalyDetector:
    def __init__(self, clip_size: int, image_size: int):
        self.clip_size = clip_size
        self.image_size = image_size

    def train(self, dir: str, batch_size: int, epochs: int = 30):
        """
        Parameters
        ----------
        dir: str
            path to dataset
        """
        self._load()
        train_set = get_video_paths(dir, ['npy'])
        train_gen = BatchTraining(train_set, batch_size, self.clip_size, self.image_size)
        self.model.fit(x=train_gen, epochs=epochs, verbose=1,
                       steps_per_epoch=int(len(train_gen.filenames) // batch_size))
        self.model.save('checkpoint.hdf5')

    def test_npy(self, path: str, checkpoint: str):
        test_set = get_video_paths(path, ['npy'])
        self.model = load_model(checkpoint, custom_objects={'LayerNormalization': LayerNormalization})
        for f in test_set:
            # show_clip(f)
            clip = np.load(f, allow_pickle=True)
            pr_clip = preprocess(clip, self.clip_size, self.image_size)
            batch = np.expand_dims(pr_clip, axis=0)
            out = self.model.predict(batch)
            cost = np.linalg.norm(np.subtract(batch, out))

            if cost > 15:
                out_clip = np.squeeze(out, axis=0)
                prev = np.expand_dims(cv2.hconcat([pr_clip[idx] for idx in range(0, self.clip_size, 3)]), axis=-1)
                after = np.expand_dims(cv2.hconcat([out_clip[idx] for idx in range(0, self.clip_size, 3)]), axis=-1)
                result = np.concatenate((prev, after), axis=0)
                cv2.imshow('prev (top), after (bottom), cost = ' + str(cost), result)
                cv2.waitKey(0)

    def _load(self):
        seq = Sequential()

        # Spatial encoder
        seq.add(TimeDistributed(Conv2D(128, (11, 11), strides=4, padding="same"),
                                input_shape=(self.clip_size, self.image_size, self.image_size, 1)))
        seq.add(LayerNormalization())
        seq.add(TimeDistributed(Conv2D(64, (5, 5), strides=2, padding="same")))
        seq.add(LayerNormalization())

        # Temporal encoder
        seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
        seq.add(LayerNormalization())
        seq.add(ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True))
        seq.add(LayerNormalization())

        # Temporal decoder
        seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
        seq.add(LayerNormalization())

        # Spatial decoder
        seq.add(TimeDistributed(Conv2DTranspose(64, (5, 5), strides=2, padding="same")))
        seq.add(LayerNormalization())
        seq.add(TimeDistributed(Conv2DTranspose(128, (11, 11), strides=4, padding="same")))
        seq.add(LayerNormalization())
        seq.add(TimeDistributed(Conv2D(1, (11, 11), activation="sigmoid", padding="same")))
        seq.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=1e-4, decay=1e-5, epsilon=1e-6))
        self.model = seq

    def _preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.image_size, self.image_size)) / 256.0
        img = np.reshape(img, (self.image_size, self.image_size, 1))
        return img
