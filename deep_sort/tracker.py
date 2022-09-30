# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
import cv2
import matplotlib.pyplot as plt
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from .nn_matching import NearestNeighborDistanceMetric
from .generate_detections import create_box_encoder
from .detection import Detection
from .preprocessing import non_max_suppression


class Tracker:
    """
    This is the multi-target tracker, which takes care of creation, keeping track, and eventual deletion of all tracks
    matching threshold = max cosine distance.

    Parameters
    ----------
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    encoder : generate_detections.ImageEncoder
        The encoder is a CNN pre-trained deep_SORT tracking model
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, max_iou_distance: float = 0.7, max_age: int = 30, n_init: int = 3):
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.metric = NearestNeighborDistanceMetric("cosine", matching_threshold=0.7)
        self.encoder = create_box_encoder('./deep_sort/mars-small128.pb', batch_size=1)

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

        # display
        cmap = plt.get_cmap('tab20b')
        self.colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    def predict(self, frame: np.ndarray, bboxes: np.ndarray, run_nms: bool = True):
        """Propagate track state distributions one time step forward and update tracks.

        Parameters
        ----------
        frame: current frame where @bboxes have been detected
        bboxes: array of shape (# objects, 6) where 6 = (x_min, y_min, x_max, y_max, score, class)
            and coordinates are global (considering the width and height of the frame)
        run_nms: if True, runs non-maxima suppression to input bboxes
        """
        boxes = bboxes[:, :4]
        scores = bboxes[:, 4]
        # classes = bboxes[:, 5]

        # encode yolo detections and feed to tracker
        features = self.encoder(frame, boxes)

        # Detection input tlwh, confidence, feature
        detections = [Detection(bbox, score, feature) for bbox, score, feature in
                      zip(boxes, scores, features)]

        if run_nms:
            # Run non-maxima suppression.
            nms_max_overlap = 0.7
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

        for track in self.tracks:
            track.predict(self.kf)

        # update
        self.update(detections)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def display(self, frame, verbose: bool = False):
        """
        Parameters
        ----------
        frame: current frame
        verbose: if True prints current tracks
        """
        for track in self.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()

            # select color
            color = self.colors[int(track.track_id) % len(self.colors)]
            color = [i * 255 for i in color]

            # bbox
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            # info bbox
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                          (int(bbox[0]) + (len(str(track.track_id))) * 17, int(bbox[1])), color, -1)
            # text
            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                        (255, 255, 255), 2)

            if verbose:
                print("Tracker ID: {}, BBox Coords (xmin, ymin, xmax, ymax): {}".format(
                str(track.track_id), (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        return frame

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1
