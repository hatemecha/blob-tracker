import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import moviepy as mpe

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
    cv2.resizeWindow('Controls', 400, 350)
    cv2.createTrackbar('Umbral', 'Controls', 244, 255, lambda x: None)

    cv2.createTrackbar('Area minima', 'Controls', 500, 5000, lambda x: None)
    cv2.createTrackbar('Distancia max', 'Controls', 50, 200, lambda x: None)
    cv2.createTrackbar('Max blobs', 'Controls', 10, 100, lambda x: None)
    cv2.createTrackbar('Historial', 'Controls', 500, 2000, lambda x: None)
    cv2.createTrackbar('Varianza', 'Controls', 16, 100, lambda x: None)
    cv2.createTrackbar('Caja B', 'Controls', 0, 255, lambda x: None)
    cv2.createTrackbar('Caja G', 'Controls', 255, 255, lambda x: None)
    cv2.createTrackbar('Caja R', 'Controls', 0, 255, lambda x: None)


def show_help_panel():
    help_img = np.zeros((220, 400, 3), dtype=np.uint8)
    lines = [
        "Deslizadores en 'Controls':",
        "Umbral - binarizacion",
        "Area minima - tamano minimo",
        "Distancia max - seguimiento",
        "Max blobs - limite objetos",
        "e: exportar  q: salir",
    ]
    for i, line in enumerate(lines):
        cv2.putText(help_img, line, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
    cv2.imshow('Ayuda', help_img)


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
    def __init__(self, max_dist): self.max_dist, self.next_id, self.objects = max_dist, 1, {}
    def update(self, detections, max_blobs):
        tracks, new = [], []
        dets = sorted(detections, key=lambda d: d['bbox'][2]*d['bbox'][3], reverse=True)[:max_blobs]
        for det in dets:
            c=det['centroid']; assigned=False
            for oid, ocent in self.objects.copy().items():
                if np.linalg.norm(np.array(ocent)-np.array(c))<=self.max_dist:
                    self.objects[oid]=c; det['id']=oid; tracks.append(det); assigned=True; break
            if not assigned: new.append(det)
        for det in new:
            oid=self.next_id; self.objects[oid]=det['centroid']; det['id']=oid; tracks.append(det); self.next_id+=1
        return tracks

class Visualizer:
    @staticmethod
    def draw(frame, tracks, color):
        for t in tracks:
            x,y,w,h=t['bbox']; oid=t['id']
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,f"ID {oid}",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
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
    cv2.namedWindow('Preview', cv2.WINDOW_NORMAL)

    pre = Preprocessor(500,16); det=BlobDetector(500); trk=Tracker(50)
    exporting = False; writer = None; bar = None

    while True:
        ret, frame = vs.read()
        if not ret: vs.reset(); continue
        thresh=cv2.getTrackbarPos('Umbral','Controls')

        det.min_area=cv2.getTrackbarPos('Area minima','Controls')
        trk.max_dist=cv2.getTrackbarPos('Distancia max','Controls')
        max_blobs=cv2.getTrackbarPos('Max blobs','Controls')
        history=cv2.getTrackbarPos('Historial','Controls')
        var_t=cv2.getTrackbarPos('Varianza','Controls')
        color=(cv2.getTrackbarPos('Caja B','Controls'),
               cv2.getTrackbarPos('Caja G','Controls'),
               cv2.getTrackbarPos('Caja R','Controls'))

        pre.update(history,var_t)
        fg,clean=pre.apply(frame,thresh)
        dets=det.detect(clean)
        tracks=trk.update(dets,max_blobs)
        out=Visualizer.draw(frame.copy(),tracks,color)
        top=np.hstack([frame,cv2.cvtColor(fg,cv2.COLOR_GRAY2BGR)])
        bottom=np.hstack([cv2.cvtColor(clean,cv2.COLOR_GRAY2BGR),out])
        mosaic=np.vstack([top,bottom])
        mosaic=cv2.resize(mosaic,(w*2,h*2))
        cv2.putText(mosaic,'Original',(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
        cv2.putText(mosaic,'Mascara FG',(w+10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
        cv2.putText(mosaic,'Mascara limpia',(10,h+20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
        cv2.putText(mosaic,'Salida',(w+10,h+20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
        show_help_panel()

        cv2.imshow('Preview',mosaic)
        key=cv2.waitKey(1)&0xFF
        if key==ord('e') and not exporting:
            exporting=True
            writer=cv2.VideoWriter(out_frames, cv2.VideoWriter_fourcc(*'mp4v'), vs.fps, (w,h))
            vs.reset()
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
                vs.reset()
        if key==ord('q'): break
    vs.release()
    cv2.destroyAllWindows()
