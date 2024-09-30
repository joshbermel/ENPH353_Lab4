#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import sys
import numpy as np

class My_App(QtWidgets.QMainWindow):

    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        self._cam_id = 0
        self._cam_fps = 10
        self._is_cam_enabled = False
        self._is_template_loaded = False
        self.sift = cv2.SIFT_create()  # Initialize SIFT

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240)

        # Timer used to trigger the camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(1000 / self._cam_fps)

        # Brute-Force matcher for SIFT
        self.bf = cv2.BFMatcher()

        self.template_image = None  # Placeholder for the template image
        self.kp_template = None
        self.des_template = None

    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)

        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]
            self.template_image = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)

            if self.template_image is not None:
                self._is_template_loaded = True

                # Run SIFT on the template image
                self.kp_template, self.des_template = self.sift.detectAndCompute(self.template_image, None)

                pixmap = QtGui.QPixmap(self.template_path)
                self.template_label.setPixmap(pixmap)

                print("Loaded template image file: " + self.template_path)
            else:
                print("Error loading template image.")

    # Source: stackoverflow.com/questions/34232632/
    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, 
                             bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    def SLOT_query_camera(self):
        # Capture a frame from the camera
        ret, frame = self._camera_device.read()

        if ret and self._is_template_loaded:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Find keypoints and descriptors in both images
            kp_frame, desc_frame = self.sift.detectAndCompute(frame_gray, None)

            # Use FLANN-based matcher to match descriptors
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(self.des_template, desc_frame, k=2)

            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            # Find homography if enough good matches
            if len(good_matches) > 10:
                src_pts = np.float32([self.kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Compute homography using RANSAC
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                # Get the template image shape
                h, w = self.template_image.shape

                # Define points to draw bounding box around detected object
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                # Draw the bounding box on the frame
                frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

            # Convert the frame with keypoints to a QPixmap and show it in the label
            pixmap = self.convert_cv_to_pixmap(frame)
            self.live_image_label.setPixmap(pixmap)

    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())

