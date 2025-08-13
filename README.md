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
  * **Object Tracking:** A proximity-based tracking system assigns unique IDs to objects and tracks them across frames.
  * **Real-time Control Panel:** Allows dynamic adjustment of key parameters such as threshold, minimum blob area, maximum tracking distance, maximum number of blobs, background subtractor history, variance threshold, and bounding box color.
  * **Mosaic Visualization:** Simultaneously displays the original frame, foreground mask, clean mask, and the output frame with tracked objects.
  * **Video Export:** Exports the processed video to a new MP4 file, complete with a progress bar.
  * **Audio Merging:** Automatically combines the audio from the original video with the exported video.

## 3\. Requirements

Ensure you have the following libraries installed. You can install them using pip:

```bash
pip install opencv-python numpy moviepy tqdm
```

`numpy` and `tqdm` are essential. `moviepy` is required for audio merging.

## 4\. Usage

1.  **Run the Script:**
    ```bash
    python blob_tracker.py
    ```
2.  **Select Video:** A file dialog will pop up for you to select your input video file (`.mp4` or `.avi`).
3.  **Select Save File:** Another file dialog will prompt you to specify the output MP4 file path and name for the processed frames.
4.  **Control Panel:** A window named "Controls" will appear with sliders to adjust parameters.
      * **Umbral:** Binary threshold for the foreground mask.
      * **AreaMin:** Minimum area (in square pixels) for a contour to be considered a blob.
      * **DistMax:** Maximum distance (in pixels) between a detected blob's centroid and an existing tracked object for it to be considered the same object.
      * **MaxBlobs:** Maximum number of blobs to track (larger blobs are prioritized).
      * **Historial:** Length of history for the MOG2 background subtraction algorithm.
      * **Varianza:** Variance threshold for the MOG2 background subtraction algorithm.
      * **CajaB, CajaG, CajaR:** BGR components of the bounding box and object ID color.
5.  **Preview Window:** The "Preview" window will display the mosaic visualization of the processing output with quadrant labels (Original, Mascara FG, Mascara limpia y Salida).
6.  **Export Video:** Press the `e` key while the "Preview" window is active to start video export. A progress bar will appear in the console. Once frame export is complete, audio will be merged with the video. The final output file will have `_with_audio.mp4` appended to its name.
7.  **Quit:** Press the `q` key to close all windows and exit the program.

## 5\. Code Structure

The script is organized into several functions and classes to modularize the different stages of video processing.

  * **`select_file()`, `select_save_file()`:** Helper functions for opening file selection dialogs using `tkinter`.
  * **`create_control_panel()`:** Sets up the OpenCV control window with sliders for all adjustable parameters.
  * **`VideoSource`:** Class to encapsulate video file reading and management.
  * **`Preprocessor`:** Class responsible for background subtraction and image preprocessing.
  * **`BlobDetector`:** Class for detecting blobs (contours) in the preprocessed mask.
  * **`Tracker`:** Class that manages the tracking of multiple objects, assigning and updating object IDs.
  * **`Visualizer`:** Static class for drawing tracking results on the frame.
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

  * **`__init__(self, max_dist)`:** Initializes the tracker with the maximum assignment distance, the next available object ID, and a dictionary of tracked objects (`id: centroid`).
  * **`update(self, detections, max_blobs)`:** Assigns detections to existing tracked objects or creates new ones. Prioritizes larger blobs up to `max_blobs`. Returns a list of tracked objects (detections with assigned IDs).

### `Visualizer`

  * **`draw(frame, tracks, color)` (static method):** Draws bounding boxes and IDs for each tracked object on the frame.

## 7\. Control Panel

The "Controls" panel allows for real-time manipulation of the following parameters:

  * **Umbral:** Used in `cv2.threshold` to binarize the foreground mask.
  * **AreaMin:** Used in `BlobDetector` to filter small contours.
  * **DistMax:** Used in `Tracker` to determine if a detected blob corresponds to an existing tracked object.
  * **MaxBlobs:** Limits the number of blobs tracked simultaneously.
  * **Historial:** The `history` parameter for `cv2.createBackgroundSubtractorMOG2`. How many frames are used for the background model.
  * **Varianza:** The `varThreshold` parameter for `cv2.createBackgroundSubtractorMOG2`. Determines how far a pixel can be from the mean to be considered foreground.
  * **CajaB, CajaG, CajaR:** BGR values for the color of the drawn bounding boxes.

## 8\. Video Export

Upon pressing `e`:

1.  The program rewinds the input video to the beginning.
2.  Initializes an `cv2.VideoWriter` object to save the processed frames.
3.  Starts a loop to process and write each frame.
4.  A `tqdm` progress bar shows the export progress.
5.  Once all frames have been written, the `VideoWriter` is released.
6.  `moviepy` is used to load the temporary video (without audio) and the audio from the original video.
7.  The audio is merged with the video, and the final result is saved with `_with_audio.mp4` appended to its filename.