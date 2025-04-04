import multiprocessing.connection
import os
from memory_stack import MemoryStack
import socket
import multiprocessing
import pickle
import struct
import sys
import time
import utils
from jetcam.csi_camera import CSICamera
import tensorflow as tf

HOST = "127.0.0.1"
PORT = 8089

jetbot = None


def predict(camera, queue: multiprocessing.Queue):
    memory_stack_size = 8
    global jetbot
    memory_stack = MemoryStack(memory_stack_size)

    jetbot = utils.init_jetbot()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, os.path.pardir, "model.h5")
    model = utils.load_model(
        model_path, memory_stack_size, model_version=utils.ModelVersion.BCNetLSTM
    )

    prev_time = time.time()
    index = 0
    while True:
        frame = camera.value
        preprocessed_stack = memory_stack.push(frame)
        input_image = preprocessed_stack.reshape(1, memory_stack_size, 100, 200, 1)
        prediction = model.predict(
            input_image,
            verbose=0,
        )

        prediction = prediction[0]
        memory_stack.push_history(prediction)

        data = pickle.dumps(frame)  ### new code
        queue.put(
            struct.pack("!L", len(data))
            + data
            + struct.pack("!fff", prediction[0], prediction[1], time.time())
        )

        if jetbot is not None:
            jetbot.set_motors(
                *utils.calculate_motor_speeds(prediction[0], prediction[1])
            )
        else:
            pass
        current_time = time.time()
        if (current_time - prev_time) < (1 / 4):
            time.sleep((1 / 4) - (current_time - prev_time))
        current_time = time.time()
        print("fps:", 1 / (current_time - prev_time))
        prev_time = current_time
        index += 1


def start_prediction(queue: multiprocessing.Queue):
    camera = CSICamera(
        width=400, height=400, capture_width=1640, capture_height=1232, capture_fps=30
    )
    camera.running = True
    predict(camera=camera, queue=queue)


def sender_process_handle(queue: multiprocessing.Queue):
    clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientsocket.connect((HOST, PORT))

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


if __name__ == "__main__":
    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=400)
            ],  # Adjust as needed
        )
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
