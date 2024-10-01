from gstreamer_pipeline import gstreamer_pipeline
import atexit
import cv2
import threading

"""
The Camera class is used to create an instance of the camera, also providing functions to take a picture or a video.
"""
class Camera():
    
    value = None
    
    def __init__(self, save_recording=True, *args, **kwargs):
        try:
#           Make sure to use the correct gstreamer pipeline. More info in gstreamer_pipeline.py
            self.save_recording=save_recording
            self._gst_str = gstreamer_pipeline()
            self.cap = cv2.VideoCapture(self._gst_str, cv2.CAP_GSTREAMER)
            if save_recording:
#               If you capture a video, you can change the target file name and frame size here.
                self.out = cv2.VideoWriter('recording.avi', cv2.VideoWriter_fourcc(*"MJPG"), 10, (816,616))
            
#           We capture 10 frames to ramp up the camera. The first couple pictures are much darker, because the camera has to get used to the light first.
            ramp_frames = 10
            for i in range(ramp_frames):
                re, ramp = self.cap.read()
                print(re)
                if not re:
                    raise RuntimeError('Could not read image from camera.')
                self.value = ramp
        except:
            self.stop()
            raise RuntimeError(
                'Could not initialize camera.  Please see error trace.')

        atexit.register(self.stop)

    def _capture_frames(self):
        while True:
            re, image = self.cap.read()
            if re:
                self.value = image
                if self.save_recording:
                    self.out.write(image)
            else:
                break
                
    def _capture_highlighted_frames(self):
        while True:
            re, image = self.cap.read()
            if re:
                if self.save_recording:
                    self.out.write(image)
            else:
                break
        
    def _capture_frame(self):
        re, image = self.cap.read()
        if re:
            self.value = image
            cv2.imwrite('pics/picture.png', img)
    
#   Run this function to capture a video.
    def capture_video(self):
        if not self.cap.isOpened():
            self.cap.open(self._gst_str, cv2.CAP_GSTREAMER)
        if not hasattr(self, 'thread') or not self.thread.isAlive():
            self.thread = threading.Thread(target=self._capture_frames)
            self.thread.start()
            
#   Run this function to capture a video in which recognized obstacles are highlighted.
    def capture_highlighted_video(self):
        if not self.cap.isOpened():
            self.cap.open(self._gst_str, cv2.CAP_GSTREAMER)
        if not hasattr(self, 'thread') or not self.thread.isAlive():
            self.thread = threading.Thread(target=self._capture_highlighted_frames)
            self.thread.start()
            
#   Run this function to take a single picture.
    def take_picture(self):
        if not self.cap.isOpened():
            self.cap.open(self._gst_str, cv2.CAP_GSTREAMER)
            self._capture_frame()
            
#   Run this function to correctly stop the camera. Is automatically called when scripts using the Camera end.
    def stop(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'thread'):
            self.thread.join()
        if hasattr(self, 'out'):
            self.out.release()
        
        