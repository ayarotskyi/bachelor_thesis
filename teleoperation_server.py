import socket
import struct
import threading
import datetime
import numpy as np

import cv2
from Adafruit_MotorHAT import Adafruit_MotorHAT
from gstreamer_pipeline import gstreamer_pipeline
from motor import Motor
from jetbot import JetBot

HOST='192.168.0.134'
PORT=8090

def server():
    i2c_bus = 1
    left_channel = 1
    right_channel = 2
    driver = Adafruit_MotorHAT(i2c_bus=i2c_bus)
    
    left_motor = Motor(driver, left_channel)
    right_motor = Motor(driver, right_channel)
    
    jetbot = JetBot(left_motor, right_motor, save_recording=False)

    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    print('Socket created')
    
    s.bind((HOST,PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')
    
    conn,addr=s.accept()
    print('Connection accepted')

    motor_lock = threading.Lock()
    left_motor_value = 0
    right_motor_value = 0

    isRecordingStarted = False

    def startRecording():
        date = datetime.datetime.now()
        frameIndex = 0
        dtype = np.dtype([('leftMotorValue', np.float32), ('rightMotorValue', np.float32)])
        motor_values = np.empty(dtype=dtype)
        def capture_frames(cap):
            try:
                while cap.isOpened(): 
                    re, frame = cap.read()
                    cv2.imwrite('outputs/{date}/images/{frameIndex}.jpg', frame)
                    with motor_lock:
                        motor_values[frameIndex] = (left_motor_value, right_motor_value)
                    frameIndex = frameIndex + 1
                    
            except KeyboardInterrupt:
                print("Data is saved under outputs/{date}")
                cap.release()
    
        def take_video():
            commandString = gstreamer_pipeline()
            cap = cv2.VideoCapture(commandString, cv2.CAP_GSTREAMER)
    
            ramp_frames = 10
            for i in range(ramp_frames):
                ret, ramp = cap.read()
            thread = threading.Thread(target=capture_frames(cap))
            
        take_video()

    while True:
        data = conn.recv(6)
        if not data:
            break  # Connection closed
        if not isRecordingStarted:
            startRecording()
        
        # Unpack the received data
        _, axis, value = struct.unpack('!BBf', data)

        print(axis, value)
        with motor_lock:
            if axis == 0:
                jetbot.left_motor.setSpeed(value)
                left_motor_value = value
            elif axis == 1:
                jetbot.right_motor.setSpeed(value)
                right_motor_value = value