
import numpy as np
import cv2
from features import get_features_frame
from scipy.ndimage.measurements import label
import time


class FrameClassificator():

    def __init__(self, frame_shape, settings,
                 svm, dt, x_scaler, x_pca,
                 scales=[],
                 ystart=[], ystop=[],
                 convert_color=False,
                 draw_debug=False):

        self.frame_shape = frame_shape
        self.convert_color = convert_color
        self.heatmap = np.zeros(frame_shape)

        self.do_frame = 0
        self.draw_debug = draw_debug

        self.frame_duration = []
        self.feature_generation = []
        self.predicting = []

        self.settings = settings

        self.svm = svm
        self.dt = dt
        self.x_scaler = x_scaler
        self.x_pca = x_pca

        self.scales = list(zip(scales, ystart, ystop))

        self.ystart = ystart
        self.ystop = ystop

    def __apply_threshold(self, threshold):
        # Zero out pixels below the threshold
        self.heatmap[self.heatmap <= threshold] = 0

    def __draw_labeled_bboxes(self, frame, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                    (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(frame, bbox[0], bbox[1], (0, 0, 255), 6)
        # Return the image
        return frame

    def ___combine_images(self, image, image1=None, image2=None, image3=None):
        """Append optional images below original input image."""
        small_shape = cv2.resize(image, (0, 0), fx=1/3, fy=1/3).shape
        if image1 is None:
            image1 = np.zeros(small_shape)
        else:
            image1 = cv2.resize(image1, small_shape[1::-1])
            if len(image1.shape) < 3:
                image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)*255
        if image2 is None:
            image2 = np.zeros(small_shape)
        else:
            image2 = cv2.resize(image2, small_shape[1::-1])
            if len(image2.shape) < 3:
                image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)*255
        if image3 is None:
            image3 = np.zeros(small_shape)
        else:
            image3 = cv2.resize(image3, small_shape[1::-1])
            if len(image3.shape) < 3:
                image3 = cv2.cvtColor(image3, cv2.COLOR_GRAY2RGB)*255

        image_above = np.concatenate((image1, image2), axis=1)
        image_below = np.concatenate((image_above, image3), axis=1)
        image_below = image_below[:, :1280, :]
        cv2.line(image_below, (small_shape[1], 0),
                 (small_shape[1], small_shape[0]),
                 (255, 255, 0))
        cv2.line(image_below, (small_shape[1]*2, 0),
                 (small_shape[1]*2, small_shape[0]),
                 (255, 255, 0))
        image = np.concatenate((image, image_below), axis=0)
        return cv2.resize(image, (0, 0), fx=0.9, fy=0.9).astype(np.uint8)

    def classify_frame(self, frame):
        self.heatmap = self.heatmap*0.7
        feature_generation = 0
        predicting = 0
        if self.convert_color:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.do_frame == 0:
            self.do_frame += 1
            t1 = time.time()

            for scale in self.scales:
                self.frame_boxes = np.copy(frame)
                tf1 = time.time()
                features = get_features_frame(frame, scale[0], scale[1], scale[2],
                                              self.settings)
                tf2 = time.time()
                feature_generation += tf2-tf1

                for feature in features:
                    heat = 0
                    top_left = feature[0]
                    bottom_right = feature[1]
                    x = feature[2]

                    tp1 = time.time()
                    x_scaled = self.x_scaler.transform([x])
                    if self.x_pca is not None:
                        x_pca = self.x_pca.transform(x_scaled)
                    else:
                        x_pca = x_scaled

                    if self.svm is not None:
                        y_svm = self.svm.predict(x_pca)
                    else:
                        y_svm = 0
                    if y_svm == 1:
                        heat += 1
                        cv2.rectangle(self.frame_boxes, top_left, bottom_right,
                                      (0, 0, 255), 6)

                        if self.dt is not None:
                            y_dt = self.dt.predict(x_scaled)
                        else:
                            y_dt = 0
                        if y_dt == 1:
                            heat += 0.4
                            cv2.rectangle(self.frame_boxes, top_left, bottom_right,
                                          (0, 255, 0), 6)
                    tp2 = time.time()
                    predicting += tp2-tp1
                    self.heatmap[top_left[1]:bottom_right[1],
                                 top_left[0]:bottom_right[0]] += heat

                """
                cv2.line(frame, (0, scale[1]), (self.frame_shape[1], scale[1]),
                         (255, 255, 0))
                cv2.line(frame, (0, scale[2]), (self.frame_shape[1], scale[2]),
                         (255, 255, 0))
                """
            t2 = time.time()
            self.frame_duration.append(t2-t1)
        else:
            if self.do_frame > 1:
                self.do_frame = 0
            else:
                self.do_frame += 1
            self.frame_duration.append(0)

        self.__apply_threshold(1)
        labels = label(self.heatmap)
        frame = self.__draw_labeled_bboxes(frame, labels)
        self.feature_generation.append(feature_generation)
        self.predicting.append(predicting)
        seconds_per_frame = sum(self.frame_duration)/len(self.frame_duration)
        frames_per_second = 1/seconds_per_frame

        cv2.putText(frame,
                    'Average frames per second: ' +
                    str(round(frames_per_second, 2)),
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2)
        cv2.putText(frame,
                    'Time for feature generation: ' +
                    str(round(sum(self.feature_generation)/len(self.feature_generation), 3)) +
                    's',
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2)
        cv2.putText(frame,
                    'Time for predictions: ' +
                    str(round(sum(self.predicting)/len(self.predicting), 3)) + 's',
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2)

        if self.draw_debug:
            frame = self.___combine_images(frame,
                                           self.frame_boxes,
                                           self.heatmap.astype(np.float32),
                                           labels[0].astype(np.uint8))
        if self.convert_color:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
