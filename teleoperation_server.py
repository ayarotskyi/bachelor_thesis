import socket
import struct
import threading
import datetime
import numpy as np
import os
import math

import cv2


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
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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

    def startRecording(cap, run_event):
        def capture_frames(cap, event):
            motor_values = np.array([], dtype=[('left', np.float64), ('right', np.float64)])
            date = str(datetime.datetime.now().timestamp())
            frameIndex = 0
            if not os.path.exists("outputs/"+date+"/images"):
                os.makedirs("outputs/"+date+"/images")
            try:
                while cap.isOpened() and event.is_set(): 
                    re, frame = cap.read()
                    cv2.imwrite('outputs/'+date+'/images/+'+str(frameIndex)+'.jpg', frame)
                    with motor_lock:
                        motor_values = np.append(motor_values, np.array([(x_axis_value, y_axis_value)], dtype=motor_values.dtype))
                    frameIndex = frameIndex + 1
            finally:
                print('Data is saved under outputs/'+date)
                cap.release()
                np.save('outputs/'+date+'/motor_values.npy', motor_values)
    
        def take_video():
                
            print("starting recording")
            thread = threading.Thread(target=capture_frames, args=(cap, run_event))
            thread.start()
            return thread
            
        return take_video()

    recording_thread = None
    run_event = threading.Event()
    run_event.set()
    if MOCK_SERVER:
        cap = cv2.VideoCapture(1)
    else:
        pipeline = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! appsink"
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    ramp_frames = 10
    for i in range(ramp_frames):
        ret, ramp = cap.read()
    try:
        while True:
            data = conn.recv(6)
            if not data:
                break  # Connection closed
            if not isRecordingStarted:
                

                recording_thread = startRecording(cap, run_event)
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
                    jetbot.set_motors(*calculate_motor_speeds(x_axis_value, y_axis_value))
    except KeyboardInterrupt: 
        run_event.clear()
        if recording_thread: recording_thread.join()
        conn.close()

def calculate_motor_speeds(x, y):
    rotation_quotient = 0.5
    # Calculate left and right motor powers
    left_power = -y + x * rotation_quotient
    right_power = -y - x * rotation_quotient

    # Normalize powers to ensure they're within -1 to 1 range
    max_power = max(abs(left_power), abs(right_power), 1)
    left_power /= max_power
    right_power /= max_power
    
    return left_power, right_power

if __name__ == '__main__':
    server()