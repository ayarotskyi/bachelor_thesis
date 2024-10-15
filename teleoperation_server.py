import socket
import struct
import threading
import datetime
import numpy as np
import os
import math

import cv2
from gstreamer_pipeline import gstreamer_pipeline


MOCK_SERVER = True

HOST='127.0.0.1'
PORT=8090

def server():
    i2c_bus = 1
    left_channel = 1
    right_channel = 2
    if MOCK_SERVER:
        jetbot = None
    else:
        from motor import Motor
        from jetbot import JetBot
        from Adafruit_MotorHAT import Adafruit_MotorHAT
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
    x_axis_value = 0
    y_axis_value = 0

    isRecordingStarted = False

    def startRecording():
        motor_values = np.array([])
        def capture_frames(cap):
            date = str(datetime.datetime.now().timestamp())
            frameIndex = 0
            if not os.path.exists("outputs/"+date+"/images"):
                os.makedirs("outputs/"+date+"/images")
            try:
                while cap.isOpened(): 
                    re, frame = cap.read()
                    cv2.imwrite('outputs/'+date+'/images/+'+str(frameIndex)+'.jpg', frame)
                    with motor_lock:
                        np.append(motor_values, [(x_axis_value, y_axis_value)])
                    frameIndex = frameIndex + 1
                    
            except KeyboardInterrupt:
                print("Data is saved under outputs/{date}")
                cap.release()
                np.save('outputs/'+date+'/motor_values.npy', motor_values)
    
        def take_video():
            if MOCK_SERVER:
                cap = cv2.VideoCapture()
            else:
                commandString = gstreamer_pipeline()
                cap = cv2.VideoCapture(commandString, cv2.CAP_GSTREAMER)
    
            ramp_frames = 10
            for i in range(ramp_frames):
                ret, ramp = cap.read()
            print("starting recording")
            thread = threading.Thread(target=capture_frames, args=(cap, ))
            thread.start()
            
        take_video()

    while True:
        try:
            data = conn.recv(6)
            if not data:
                break  # Connection closed
            if not isRecordingStarted:
                startRecording()
                isRecordingStarted = True
            # Unpack the received data
            _, axis, value = struct.unpack('!BBf', data)

            with motor_lock:
                if axis == 0:
                    x_axis_value = value
                elif axis == 1:
                    y_axis_value = value
                if MOCK_SERVER:
                    print(calculate_motor_speeds(x_axis_value, y_axis_value))
                else: 
                    jetbot.set_motors(**calculate_motor_speeds(x_axis_value, y_axis_value))
        finally:
            break

def calculate_motor_speeds(x, y):
    """
    Calculate motor speeds based on controller input.
    x: left/right axis value (-1 to 1)
    y: up/down axis value (-1 to 1)
    Returns: (left_speed, right_speed)
    """
    # Convert x, y to polar coordinates
    r = math.sqrt(x*x + y*y)
    theta = math.atan2(y, x)

    # Calculate left and right motor speeds
    left = r * math.sin(theta + math.pi/4)
    right = r * math.sin(theta - math.pi/4)

    # Normalize to [-1, 1]
    max_value = max(abs(left), abs(right), 1)
    left /= max_value
    right /= max_value

    return left, right

if __name__ == '__main__':
    server()