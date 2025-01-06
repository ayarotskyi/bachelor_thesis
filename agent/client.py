import tensorflow as tf
import os
import cv2
from memory_stack import MemoryStack
import socket
import threading
import pickle
import struct
import time

HOST = '127.0.0.1'
PORT = 8089
MOCK_JETBOT = True
STREAM_VIDEO = False
FINISHED = False

global_image = None
global_prediction = None

def load_model(model_path: str) -> tf.keras.Model:
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    try:
        # Load the model
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")
    
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

def predict(cap):
    global global_image, global_prediction, FINISHED
    memory_stack = MemoryStack()

    if MOCK_JETBOT:
        jetbot = None
    else:
        i2c_bus = 1
        left_channel = 1
        right_channel = 2
        from motor import Motor
        from jetbot import JetBot
        from Adafruit_MotorHAT import Adafruit_MotorHAT
        driver = Adafruit_MotorHAT(i2c_bus=i2c_bus)
        left_motor = Motor(driver, left_channel)
        right_motor = Motor(driver, right_channel)
        jetbot = JetBot(left_motor, right_motor, save_recording=False)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, os.path.pardir, 'model.h5')
    model = load_model(model_path)

    # current_time = time.time_ns()
    while cap.isOpened(): 
        re, frame = cap.read()
        preprocessed_stack = memory_stack.push(frame)
        input_image = preprocessed_stack.reshape(1, 400, 400, 1)
        prediction = model.predict(input_image, verbose=0)

        global_image = frame
        global_prediction = prediction[0]
        if jetbot is not None:
            jetbot.set_motors(*calculate_motor_speeds(global_prediction[0], global_prediction[1]))
        else:
            print("Predicted joystick position: ", calculate_motor_speeds(global_prediction[0], global_prediction[1]))
        
        # updated_time = time.time_ns()
        # time_delta = updated_time - current_time
        # current_time = updated_time
        # print("fps: " + str( 1_000_000_000 / time_delta))
    FINISHED = True

def start_prediction_thread():
    pipeline = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! appsink"
    cap = cv2.VideoCapture(0)
    ramp_frames = 10
    for i in range(ramp_frames):
        ret, ramp = cap.read()
    thread = threading.Thread(target=predict, args=(cap,))
    return thread, cap

def send_video_frames():
    global global_image, global_prediction

    clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    clientsocket.connect((HOST,PORT))
    clientsocket.send(b"a" if STREAM_VIDEO else b"b")

    while global_image is None or global_prediction is None:
        continue
    while STREAM_VIDEO:
        data = pickle.dumps(global_image) ### new code
        clientsocket.sendall(struct.pack("L", len(data))+data+struct.pack("ff", global_prediction[0], global_prediction[1]))
    if not STREAM_VIDEO:
        while not FINISHED:
            time.sleep(1)
        
def start_socket_thread():
    thread = threading.Thread(target=send_video_frames)
    return thread

if __name__ == '__main__':
    try:
        prediction_thread, cap = start_prediction_thread()
        prediction_thread.start()
        socket_thread = start_socket_thread()
        socket_thread.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cap.release()