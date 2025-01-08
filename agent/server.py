import socket
import cv2
import pickle
import struct
from client import calculate_motor_speeds
import numpy as np
import os
import datetime
import multiprocessing

# only works with python3

# Change it to the port of your computer (it should be connected to the same network as the JetBot)
HOST='127.0.0.1'
PORT=8089

run_data = np.array([], dtype=[('x', np.float64), ('y', np.float64), ('timestamp', np.float64)])
date = str(int(datetime.datetime.now().timestamp() * 1000))
directory = "results/"+date

def image_saver_handle(directory, image_data, index):
    frame = pickle.loads(image_data)
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(directory+"/"+str(index)+".jpg", frame)


def receive_data(process_pool, conn: socket.socket):
    global run_data

    data = b'' ### CHANGED
    payload_size = struct.calcsize("L") ### CHANGED
    predictions_size = struct.calcsize("fff")

    index = 0

    while True:

        # Retrieve message size
        while len(data) < payload_size:
            data += conn.recv(4096)

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0] ### CHANGED

        # Retrieve all data based on message size
        while len(data) < msg_size:
            data += conn.recv(4096)

        frame_data = data[:msg_size]
        data = data[msg_size:]

        while len(data) < predictions_size:
            data += conn.recv(4096)

        packed_predictions = data[:predictions_size]
        data = data[predictions_size:]
        x, y, timestamp = struct.unpack("fff", packed_predictions)

        # Extract frame
        frame = pickle.loads(frame_data)

        process_pool.apply_async(image_saver_handle, args=(directory, frame_data, index))
        index += 1

        # Update data array
        run_data = np.append(run_data, np.array([(x, y, timestamp)], dtype=run_data.dtype))

        # Draw controls line
        x, y = calculate_motor_speeds(x, y)
        height, width = frame.shape[:2]
        rotated_x = 0.5-y
        rotated_y = x 
        scaled_x = int(rotated_x * height)
        scaled_y = int(rotated_y * width / 2)
        start_x = width // 2
        start_y = height - 1
        cv2.line(frame, (start_x, start_y), (start_x - scaled_x, start_y - scaled_y), (0,255,0), 2)

        # Display
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

# Runs the socket server which displays the video when a client connects
# `python3 stream_video.py server`
if __name__ == '__main__':
    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    print('Socket created')
    
    s.bind((HOST,PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')
    
    conn,addr=s.accept()
    print('Connection accepted')

    try:
        process_pool = multiprocessing.Pool(2)
        receive_data(process_pool, conn)
    except:
        process_pool.close()
        process_pool.join()

        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save(directory+"/data.npy", run_data[['x', 'y', 'timestamp']])