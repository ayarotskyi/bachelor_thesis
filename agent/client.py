import multiprocessing.connection
import keras
import os
import cv2
from memory_stack import MemoryStack
import socket
import multiprocessing
import pickle
import struct
import sys
import time

HOST = '127.0.0.1'
PORT = 8089
MOCK_JETBOT = True


def load_model(model_path: str) -> keras.Model:
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    try:
        # Load the model
        model = keras.models.load_model(model_path)
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

def predict(cap, queue: multiprocessing.Queue):
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

        prediction = prediction[0]
        data = pickle.dumps(frame) ### new code
        queue.put(struct.pack("!L", len(data))+data+struct.pack("!fff", prediction[0], prediction[1], time.time()))

        if jetbot is not None:
            jetbot.set_motors(*calculate_motor_speeds(prediction[0], prediction[1]))
        else:
            print("Predicted joystick position: ", calculate_motor_speeds(prediction[0], prediction[1]))
        
        # updated_time = time.time_ns()
        # time_delta = updated_time - current_time
        # current_time = updated_time
        # print("fps: " + str( 1_000_000_000 / time_delta))

def start_prediction(queue: multiprocessing.Queue):
    pipeline = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! appsink"
    cap = cv2.VideoCapture(0)
    ramp_frames = 10
    for i in range(ramp_frames):
        ret, ramp = cap.read()
    predict(cap=cap, queue=queue)

def sender_process_handle(queue: multiprocessing.Queue):
    clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    clientsocket.connect((HOST,PORT))

    try:
        while True:
            data = queue.get()
            clientsocket.send(data)
    except:
        clientsocket.close()


def start_sender_process(queue: multiprocessing.Queue):
    process = multiprocessing.Process(target=sender_process_handle, args=(queue,))
    process.start()
    return process

if __name__ == '__main__':
    try:
        queue = multiprocessing.Queue()
        sender_process = start_sender_process(queue)

        start_prediction(queue=queue)
    except KeyboardInterrupt:
        # kill the process
        if sender_process.is_alive():
            sender_process.kill()
        sender_process.join()

        # Drain the queue before closing
        while not queue.empty():
            try:
                queue.get_nowait()
            except:
                break
        queue.cancel_join_thread()
        queue.close()

        sys.exit(0)