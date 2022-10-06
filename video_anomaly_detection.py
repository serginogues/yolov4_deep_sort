from anomaly_detection.network import AnomalyDetector

CLIP_SIZE = 20
IMG_SIZE = 160

DATASET = r'C:\Users\azken\Documents\yolo-deepsort\data\clips'
EPOCHS = 10
BATCH_SIZE = 1

TRAIN = False
TEST = 'data/clips_anomaly'
CKPT = 'checkpoint.hdf5'


if __name__ == '__main__':
    model = AnomalyDetector(clip_size=CLIP_SIZE, image_size=IMG_SIZE)

    if TRAIN:
        model.train(dir=DATASET, batch_size=BATCH_SIZE, epochs=EPOCHS)
    else:

        model.test_npy(TEST, CKPT)
