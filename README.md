# Blob Tracker

## Table of Contents

1.  [General Description](#1-general-description)
2.  [Features](#2-features)
3.  [Requirements](#3-requirements)
4.  [Usage](#4-usage)
5.  [Code Structure](#5-code-structure)
6.  [Core Classes](#6-core-classes)
7.  [Control Panel](#7-control-panel)
8.  [Video Export](#8-video-export)

-----

## 1\. General Description

`blob_tracker.py` is an OpenCV-based object tracking tool designed to detect, track, and visualize "blobs" (moving objects) within a video file. It utilizes background subtraction to identify foreground objects and allows the user to adjust various parameters in real-time via a control panel. The output is displayed in a preview window and can be exported to a new video file with an option to merge the original audio.

## 2\. Features

  * **Video File Selection:** Graphical interface to easily select the input video.
  * **Foreground Segmentation:** Uses the MOG2 algorithm for background subtraction to create a foreground mask.
  * **Blob Detection:** Identifies contours in the foreground mask and filters out those below a defined minimum area.
  * **Object Tracking:** Uses a cost matrix with the Hungarian algorithm to assign detections to tracked objects, with an optional Kalman filter for smoother position estimates.
  * **ROI Selection:** Define a rectangular region of interest to focus detection and tracking.
  * **Per-object Colors & Motion Trails:** Each object receives a unique color and an optional trail showing recent positions.
  * **Real-time Control Panel:** Adjust threshold, minimum blob area, maximum tracking distance, number of blobs, background subtractor history, variance threshold, Kalman filtering, and trail visualization.
  * **Preset Persistence:** Save and load control-panel settings with keyboard shortcuts.
  * **Mosaic Visualization:** Simultaneously displays the original frame, foreground mask, clean mask, and the output frame with tracked objects.
  * **Video & CSV Export:** Exports the processed video to a new MP4 file and logs tracked positions to a CSV.
  * **Audio Merging:** Automatically combines the audio from the original video with the exported video.

## 3\. Requirements

Ensure you have the following libraries installed. You can install them using pip:

```bash
pip install opencv-python numpy scipy moviepy tqdm
```

`numpy`, `scipy`, and `tqdm` are essential. `moviepy` is required for audio merging.

## 4\. Usage

1.  **Run the Script:**
    ```bash
    python blob_tracker.py
    ```
2.  **Select Video:** A file dialog will pop up for you to select your input video file (`.mp4` or `.avi`).
3.  **Select Save File:** Another file dialog will prompt you to specify the output MP4 file path and name for the processed frames.
4.  **Control Panel:** A window named "Controls" will appear with sliders to adjust parameters.
      * **Umbral:** Binary threshold for the foreground mask.

      * **Area minima:** Minimum area (in square pixels) for a contour to be considered a blob.
      * **Distancia max:** Maximum distance (in pixels) between a detected blob's centroid and an existing tracked object for it to be considered the same object.
      * **Max blobs:** Maximum number of blobs to track (larger blobs are prioritized).
      * **Kalman:** Toggle (0/1) for applying Kalman filter smoothing to tracked positions.
      * **Historial:** Length of history for the MOG2 background subtraction algorithm.
      * **Varianza:** Variance threshold for the MOG2 background subtraction algorithm.
      * **Ver rastro:** Toggle (0/1) to show motion trails.
      * **Len rastro:** Number of recent positions drawn in the trail.
5.  **Help Window:** A separate "Ayuda" window summarizes these controls and lists key commands (`s` to save, `l` to load, `r` to select/reset ROI, `e` to export, `q` to quit).
6.  **Preview Window:** The "Preview" window will display the mosaic visualization of the processing output with quadrant labels (Original, Mascara FG, Mascara limpia y Salida). The top-right corner shows the current frame and total frame count for quick progress reference.
7.  **Export Video:** Press the `e` key while the "Preview" window is active to start video export. A progress bar will appear in the console. Once frame export is complete, audio will be merged with the video. The final output file will have `_with_audio.mp4` appended to its name.
8.  **Quit:** Press the `q` key to close all windows and exit the program.
9.  **CSV Log:** Upon exit, a CSV file with tracked positions is saved alongside the video (`*_tracks.csv`).


## 5\. Code Structure

The script is organized into several functions and classes to modularize the different stages of video processing.

  * **`select_file()`, `select_save_file()`:** Helper functions for opening file selection dialogs using `tkinter`.
  * **`create_control_panel()`:** Sets up the OpenCV control window with sliders for all adjustable parameters, including a toggle for Kalman filtering.
  * **`VideoSource`:** Class to encapsulate video file reading and management.
  * **`Preprocessor`:** Class responsible for background subtraction and image preprocessing.
  * **`BlobDetector`:** Class for detecting blobs (contours) in the preprocessed mask.
  * **`Tracker`:** Class that manages the tracking of multiple objects, using the Hungarian algorithm for assignment, optional Kalman filters, and per-object colors and trails.
  * **`Visualizer`:** Static class for drawing tracking results and optional motion trails on the frame.
  * **`if __name__ == '__main__':` (Main Block):** The main loop that initializes classes, reads frames, applies processing, updates parameters from the control panel, and manages visualization and export.

## 6\. Core Classes

### `VideoSource`

  * **`__init__(self, path)`:** Initializes the OpenCV `VideoCapture` object and retrieves total frames and FPS.
  * **`read()`:** Reads the next frame from the video.
  * **`reset()`:** Resets the video to the first frame.
  * **`release()`:** Releases the `VideoCapture` object.

### `Preprocessor`

  * **`__init__(self, history, var_thresh)`:** Initializes the MOG2 background subtractor.
  * **`_reset()`:** Reinitializes the background subtractor with current parameters.
  * **`update(self, history, var_thresh)`:** Updates the background subtractor parameters if they've changed and resets it if necessary.
  * **`apply(self, frame, thresh)`:** Applies background subtraction, binary thresholding, and morphological operations (opening) to the frame. Returns the foreground mask and the clean mask.

### `BlobDetector`

  * **`__init__(self, min_area)`:** Initializes the detector with the minimum blob area.
  * **`detect(self, mask)`:** Finds contours in the mask and filters out those smaller than `min_area`. Returns a list of blob dictionaries with `centroid` and `bbox`.

### `Tracker`

  * **`__init__(self, max_dist, use_kalman=False)`:** Initializes the tracker with the maximum assignment distance, optional Kalman filtering, and stores object states, colors, and trails.
  * **`update(self, detections, max_blobs, trail_len)`:** Builds a cost matrix between detections and existing objects and uses the Hungarian algorithm for assignment. New objects receive IDs and colors; trails are trimmed to `trail_len`. Returns a list of tracked detections with IDs, colors, and trails.

### `Visualizer`

  * **`draw(frame, tracks, show_trails)` (static method):** Draws per-object bounding boxes, IDs, and optionally motion trails on the frame.

## 7\. Control Panel

The "Controls" panel allows for real-time manipulation of the following parameters:

  * **Umbral:** Used in `cv2.threshold` to binarize the foreground mask.

  * **Area minima:** Used in `BlobDetector` to filter small contours.
  * **Distancia max:** Used in `Tracker` to determine if a detected blob corresponds to an existing tracked object.
  * **Max blobs:** Limits the number of blobs tracked simultaneously.
  * **Kalman:** Turns Kalman filter smoothing on (1) or off (0) for tracked positions.
  * **Historial:** The `history` parameter for `cv2.createBackgroundSubtractorMOG2`. How many frames are used for the background model.
  * **Varianza:** The `varThreshold` parameter for `cv2.createBackgroundSubtractorMOG2`. Determines how far a pixel can be from the mean to be considered foreground.
  * **Ver rastro:** Toggles the drawing of motion trails.
  * **Len rastro:** Number of points kept in each trail.


## 8\. Video Export

Upon pressing `e`:

1.  The program rewinds the input video to the beginning.
2.  Initializes an `cv2.VideoWriter` object to save the processed frames.
3.  Starts a loop to process and write each frame.
4.  A `tqdm` progress bar shows the export progress.
5.  Once all frames have been written, the `VideoWriter` is released.
6.  `moviepy` is used to load the temporary video (without audio) and the audio from the original video.
7.  The audio is merged with the video, and the final result is saved with `_with_audio.mp4` appended to its filename.