import socket
import cv2
import pickle
import struct

# only works with python3

# Change it to the port of your computer (it should be connected to the same network as the JetBot)
HOST='127.0.0.1'
PORT=8089

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
    
    data = b'' ### CHANGED
    payload_size = struct.calcsize("L") ### CHANGED
    predictions_size = struct.calcsize("ff")

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
        x, y = struct.unpack("ff", packed_predictions)
        print(x, y)

        # Extract frame
        frame = pickle.loads(frame_data)

        # Display
        cv2.imshow('frame', frame)
        cv2.waitKey(1)