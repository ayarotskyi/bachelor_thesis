import multiprocessing.connection
import os
import cv2
from memory_stack import MemoryStack
import socket
import multiprocessing
import pickle
import struct
import sys
import time
import utils

HOST = '127.0.0.1'
PORT = 8089

jetbot = None

def predict(cap, queue: multiprocessing.Queue):
    global jetbot
    memory_stack = MemoryStack()

    jetbot = utils.init_jetbot()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, os.path.pardir, 'model.h5')
    model = utils.load_model(model_path, model_version=utils.ModelVersion.LSTM)

    while cap.isOpened(): 
        re, frame = cap.read()
        preprocessed_stack = memory_stack.push(frame)
        input_image = preprocessed_stack.reshape(1, 4, 100, 400, 1)
        prediction = model.predict(input_image, verbose=0)

        prediction = prediction[0]
        data = pickle.dumps(frame) ### new code
        queue.put(struct.pack("!L", len(data))+data+struct.pack("!fff", prediction[0], prediction[1], time.time()))

        if jetbot is not None:
            jetbot.set_motors(*utils.calculate_motor_speeds(prediction[0], prediction[1]))
        else:
            pass

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
    except:
        sender_process.join()

        # Drain the queue before closing
        while not queue.empty():
            try:
                queue.get_nowait()
            except:
                break
        if jetbot:
            jetbot.stop()
        queue.cancel_join_thread()
        queue.close()

        sys.exit(0)