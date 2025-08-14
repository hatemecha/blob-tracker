import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import moviepy as mpe

from scipy.optimize import linear_sum_assignment

import json
import os
import csv


# Select file dialogs
def select_file():
    root = tk.Tk(); root.withdraw()
    path = filedialog.askopenfilename(title='Seleccionar video', filetypes=[('Video Files','*.mp4 *.avi')])
    return path or None

def select_save_file():
    root = tk.Tk(); root.withdraw()
    path = filedialog.asksaveasfilename(title='Guardar salida (frames only)', defaultextension='.mp4', filetypes=[('MP4','.mp4')])
    return path or None

# Create control panel
def create_control_panel():
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Controls', 400, 400)
    cv2.createTrackbar('Umbral', 'Controls', 244, 255, lambda x: None)

    cv2.createTrackbar('Area minima', 'Controls', 500, 5000, lambda x: None)
    cv2.createTrackbar('Distancia max', 'Controls', 50, 200, lambda x: None)
    cv2.createTrackbar('Max blobs', 'Controls', 10, 100, lambda x: None)
    cv2.createTrackbar('Historial', 'Controls', 500, 2000, lambda x: None)
    cv2.createTrackbar('Varianza', 'Controls', 16, 100, lambda x: None)
    cv2.createTrackbar('Kalman', 'Controls', 0, 1, lambda x: None)
    cv2.createTrackbar('Ver rastro', 'Controls', 1, 1, lambda x: None)
    cv2.createTrackbar('Len rastro', 'Controls', 20, 200, lambda x: None)
    cv2.createTrackbar('Color B', 'Controls', 0, 255, lambda x: None)
    cv2.createTrackbar('Color G', 'Controls', 255, 255, lambda x: None)
    cv2.createTrackbar('Color R', 'Controls', 0, 255, lambda x: None)


def show_help_panel():
    help_img = np.zeros((360, 400, 3), dtype=np.uint8)
    lines = [
        "Deslizadores en 'Controls':",
        "Umbral - binarizacion",
        "Area minima - tamano minimo",
        "Distancia max - seguimiento",
        "Max blobs - limite objetos",

        "Kalman - suavizado (0 off, 1 on)",
        "Ver rastro - mostrar rastro",
        "Len rastro - puntos en rastro",
        "Color B/G/R - color del recuadro",

        "s: guardar  l: cargar",
        "r: ROI (definir/reset)",

        "e: exportar  q: salir",
    ]
    for i, line in enumerate(lines):
        cv2.putText(help_img, line, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
    cv2.imshow('Ayuda', help_img)


# Trackbar settings persistence
SETTINGS_FILE = 'trackbar_settings.json'
TRACKBAR_NAMES = ['Umbral','Area minima','Distancia max','Max blobs','Historial','Varianza','Kalman','Ver rastro','Len rastro','Color B','Color G','Color R']


def save_trackbar_settings(path=SETTINGS_FILE):
    data = {name: cv2.getTrackbarPos(name, 'Controls') for name in TRACKBAR_NAMES}
    with open(path, 'w') as f:
        json.dump(data, f)
    print(f'Trackbar settings saved to {path}')


def load_trackbar_settings(path=SETTINGS_FILE):
    if not os.path.exists(path):
        return False
    with open(path, 'r') as f:
        data = json.load(f)
    for name, val in data.items():
        if name in TRACKBAR_NAMES:
            cv2.setTrackbarPos(name, 'Controls', int(val))
    print(f'Trackbar settings loaded from {path}')
    return True


# Core classes
class VideoSource:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
    def read(self): return self.cap.read()
    def reset(self): self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    def release(self): self.cap.release()

class Preprocessor:
    def __init__(self, history, var_thresh):
        self.history = history; self.var_thresh = var_thresh
        self._reset()
    def _reset(self):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=self.history,
                                                   varThreshold=self.var_thresh,
                                                   detectShadows=True)
    def update(self, history, var_thresh):
        if history!=self.history or var_thresh!=self.var_thresh:
            self.history, self.var_thresh = history, var_thresh
            self._reset()
    def apply(self, frame, thresh):
        fg = self.bg.apply(frame)
        _, mask = cv2.threshold(fg, thresh, 255, cv2.THRESH_BINARY)
        clean = cv2.morphologyEx(mask,
                                 cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),
                                 iterations=2)
        return fg, clean

class BlobDetector:
    def __init__(self, min_area): self.min_area = min_area
    def detect(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for cnt in contours:
            if cv2.contourArea(cnt) >= self.min_area:
                x,y,w,h = cv2.boundingRect(cnt)
                c = (x+w//2, y+h//2)
                blobs.append({'centroid':c, 'bbox':(x,y,w,h)})
        return blobs

class Tracker:
    def __init__(self, max_dist, use_kalman=False):
        self.max_dist = max_dist
        self.use_kalman = use_kalman
        self.next_id = 1
        self.objects = {}
        self.filters = {}
        self.colors = {}
        self.trails = {}

    def _create_kf(self, centroid):
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32)
        kf.statePre = np.array([[centroid[0]], [centroid[1]], [0], [0]], np.float32)
        kf.statePost = kf.statePre.copy()
        return kf

    def set_use_kalman(self, flag):
        self.use_kalman = flag
        if not flag:
            self.filters.clear()

    def update(self, detections, max_blobs, trail_len, color=(0,255,0)):
        tracks = []
        dets = sorted(detections,
                      key=lambda d: d['bbox'][2] * d['bbox'][3],
                      reverse=True)[:max_blobs]

        obj_ids = list(self.objects.keys())
        obj_centroids = np.array([self.objects[i] for i in obj_ids]) if obj_ids else np.empty((0, 2))
        det_centroids = np.array([d['centroid'] for d in dets]) if dets else np.empty((0, 2))

        if obj_centroids.size and det_centroids.size:
            cost = np.linalg.norm(obj_centroids[:, None, :] - det_centroids[None, :, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(cost)
        else:
            row_ind, col_ind = np.array([], dtype=int), np.array([], dtype=int)

        assigned_dets = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= self.max_dist:
                oid = obj_ids[r]
                det = dets[c]
                assigned_dets.add(c)
                if self.use_kalman:
                    if oid not in self.filters:
                        self.filters[oid] = self._create_kf(det['centroid'])
                    kf = self.filters[oid]
                    kf.predict()
                    measurement = np.array([[np.float32(det['centroid'][0])],
                                            [np.float32(det['centroid'][1])]])
                    estimate = kf.correct(measurement)
                    pos = (int(estimate[0]), int(estimate[1]))
                    self.objects[oid] = pos
                    det['centroid'] = pos
                else:
                    self.objects[oid] = det['centroid']
                self.trails.setdefault(oid, []).append(det['centroid'])
                self.trails[oid] = self.trails[oid][-trail_len:]
                det['id'] = oid
                self.colors[oid] = color
                det['color'] = self.colors[oid]
                det['trail'] = self.trails[oid]
                tracks.append(det)
        for idx, det in enumerate(dets):
            if idx not in assigned_dets:
                oid = self.next_id
                self.next_id += 1
                self.objects[oid] = det['centroid']
                self.colors[oid] = color
                self.trails[oid] = [det['centroid']]
                det['id'] = oid
                det['color'] = self.colors[oid]
                det['trail'] = self.trails[oid]
                if self.use_kalman:
                    self.filters[oid] = self._create_kf(det['centroid'])
                tracks.append(det)
        return tracks

class Visualizer:
    @staticmethod
    def draw(frame, tracks, show_trails):
        for t in tracks:
            x,y,w,h=t['bbox']; oid=t['id']
            color = t.get('color', (0,255,0))
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,f"ID {oid}",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
            if show_trails:
                pts = t.get('trail', [])
                if len(pts) > 1:
                    cv2.polylines(frame, [np.array(pts, np.int32)], False, color, 2)
        return frame

# Main
if __name__=='__main__':
    src = select_file()
    if not src:
        print('No video selected. Exiting.')
        exit()
    out_frames = select_save_file()
    if not out_frames:
        print('No output file selected. Exiting.')
        exit()
    vs = VideoSource(src)
    ret, frame = vs.read()
    if not ret: exit()
    h, w = frame.shape[:2]
    create_control_panel()
    load_trackbar_settings()
    cv2.namedWindow('Preview', cv2.WINDOW_NORMAL)

    pre = Preprocessor(500,16); det=BlobDetector(500); trk=Tracker(50)
    exporting = False; writer = None; bar = None
    roi = None
    frame_idx = 0
    track_logs = []

    while True:
        ret, frame = vs.read()
        if not ret:
            vs.reset(); frame_idx = 0; continue
        thresh=cv2.getTrackbarPos('Umbral','Controls')

        det.min_area=cv2.getTrackbarPos('Area minima','Controls')
        trk.max_dist=cv2.getTrackbarPos('Distancia max','Controls')
        max_blobs=cv2.getTrackbarPos('Max blobs','Controls')
        trk.set_use_kalman(bool(cv2.getTrackbarPos('Kalman','Controls')))
        history=cv2.getTrackbarPos('Historial','Controls')
        var_t=cv2.getTrackbarPos('Varianza','Controls')
        show_trails=bool(cv2.getTrackbarPos('Ver rastro','Controls'))
        trail_len=cv2.getTrackbarPos('Len rastro','Controls')
        color_b=cv2.getTrackbarPos('Color B','Controls')
        color_g=cv2.getTrackbarPos('Color G','Controls')
        color_r=cv2.getTrackbarPos('Color R','Controls')
        current_color=(color_b,color_g,color_r)

        pre.update(history,var_t)
        fg,clean=pre.apply(frame,thresh)
        if roi is not None:
            x,y,w_roi,h_roi = roi
            mask_roi = np.zeros_like(clean)
            mask_roi[y:y+h_roi, x:x+w_roi] = clean[y:y+h_roi, x:x+w_roi]
            clean = mask_roi
            mask_fg = np.zeros_like(fg)
            mask_fg[y:y+h_roi, x:x+w_roi] = fg[y:y+h_roi, x:x+w_roi]
            fg = mask_fg
        dets=det.detect(clean)
        tracks=trk.update(dets,max_blobs,trail_len,current_color)
        out=Visualizer.draw(frame.copy(),tracks,show_trails)
        if roi is not None:
            x,y,w_roi,h_roi = roi
            cv2.rectangle(out,(x,y),(x+w_roi,y+h_roi),(0,255,255),1)
        for t in tracks:
            track_logs.append([frame_idx, t['id'], t['centroid'][0], t['centroid'][1]])
        frame_idx += 1
        top=np.hstack([frame,cv2.cvtColor(fg,cv2.COLOR_GRAY2BGR)])
        bottom=np.hstack([cv2.cvtColor(clean,cv2.COLOR_GRAY2BGR),out])
        mosaic=np.vstack([top,bottom])
        mosaic=cv2.resize(mosaic,(w*2,h*2))
        cv2.putText(mosaic,'Original',(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
        cv2.putText(mosaic,'Mascara FG',(w+10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
        cv2.putText(mosaic,'Mascara limpia',(10,h+20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
        cv2.putText(mosaic,'Salida',(w+10,h+20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
        progress_text = f"Frame {frame_idx}/{vs.total_frames}"
        text_size, _ = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(mosaic, progress_text,
                    (mosaic.shape[1] - text_size[0] - 10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        show_help_panel()

        cv2.imshow('Preview',mosaic)
        key=cv2.waitKey(1)&0xFF
        if key==ord('s'):
            save_trackbar_settings()
        if key==ord('l'):
            load_trackbar_settings()
        if key==ord('r'):
            roi = cv2.selectROI('Preview', frame, fromCenter=False, showCrosshair=True)
            if roi == (0,0,0,0):
                roi = None
        if key==ord('e') and not exporting:
            exporting=True
            writer=cv2.VideoWriter(out_frames, cv2.VideoWriter_fourcc(*'mp4v'), vs.fps, (w,h))
            vs.reset(); frame_idx = 0
            bar=tqdm(total=vs.total_frames, desc='Exporting frames')
        if exporting:
            writer.write(out)
            bar.update(1)
            if bar.n>=vs.total_frames:
                writer.release(); bar.close(); exporting=False
                # Merge audio using with_audio
                temp_clip = mpe.VideoFileClip(out_frames)
                audio_clip = mpe.AudioFileClip(src)
                final = temp_clip.with_audio(audio_clip)
                final.write_videofile(out_frames.replace('.mp4','_with_audio.mp4'), codec='libx264', audio_codec='aac')
                print('Export complete with audio')
                vs.reset(); frame_idx = 0
        if key==ord('q'): break
    vs.release()
    cv2.destroyAllWindows()
    with open(out_frames.replace('.mp4','_tracks.csv'),'w',newline='') as f:
        writer_csv=csv.writer(f)
        writer_csv.writerow(['frame','id','x','y'])
        writer_csv.writerows(track_logs)
