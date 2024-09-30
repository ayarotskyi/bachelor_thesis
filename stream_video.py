import socket
import cv2
import pickle
import sys
import struct
from gstreamer_pipeline import gstreamer_pipeline
import threading

# only works with python3

# Change it to the port of your computer (it should be connected to the same network as the JetBot)
HOST='192.168.0.195'
PORT=8089

# Runs the socket server which displays the video when a client connects
# `python3 stream_video.py server`
def server():
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

        # Extract frame
        frame = pickle.loads(frame_data)

        # Display
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

# Run this on JetBot:
# `python3 stream_video.py client`
def client():
    def capture_frames(cap, clientsocket):
        try:
            while cap.isOpened(): 
                re, frame = cap.read()
                data = pickle.dumps(frame) ### new code
                clientsocket.sendall(struct.pack("L", len(data))+data)
        except KeyboardInterrupt:
            cap.release()

    def take_video():
        commandString = gstreamer_pipeline()
        cap = cv2.VideoCapture(commandString, cv2.CAP_GSTREAMER)

        ramp_frames = 10
        for i in range(ramp_frames):
            ret, ramp = cap.read()
        clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        clientsocket.connect((HOST,PORT))
        thread = threading.Thread(target=capture_frames(cap, clientsocket))
        
    take_video()

if __name__ == "__main__":
    if sys.argv[len(sys.argv) - 1] == 'client':
        client()
    else:
        server()