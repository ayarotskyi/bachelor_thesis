import pygame
import socket
import struct
import time
import random
import threading
import os

MOCK_GAMEPAD = True

SERVER_ADDRESS = "127.0.0.1"  # Change this to your server's address
SERVER_PORT = 8090  # Change this to your server's port

class GamepadSender:
    def __init__(self, server_address, server_port):
        self.server_address = server_address
        self.server_port = server_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.running = False
        self.running_lock = threading.Lock()

        # Initialize Pygame
        pygame.init()
        pygame.joystick.init()

        if MOCK_GAMEPAD:
            return

        # Check for available joysticks
        if pygame.joystick.get_count() == 0:
            raise ValueError("No joystick/gamepad found")

        # Initialize the first joystick
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        print(f"Initialized gamepad: {self.joystick.get_name()}")

    def connect(self):
        try:
            self.sock.connect((self.server_address, self.server_port))
            print(f"Connected to server at {self.server_address}:{self.server_port}")
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            return False
        return True

    def send_event(self, axis, value):
        # Pack the event data into a struct
        # Format: 
        # - B (unsigned char) for axis number (1 byte)
        # - f (float) for axis value (4 bytes)
        packed_data = struct.pack('!Bf', axis, value)
        
        self.sock.sendall(packed_data)

    def receive_file(self, path):
        filetodown = open(path, "a+b")
        data = self.sock.recv(4)
        filesize, = struct.unpack("!I", data)
        data = b""
        while filesize > len(data):
            data += self.sock.recv(filesize - len(data))
        filetodown.write(data)
        filetodown.close()

    def wait_for_data(self):
        date_data = self.sock.recv(8)
        date_timestamp, = struct.unpack("!Q", date_data)
        date = str(date_timestamp)
        if not os.path.exists("outputs/"+date+"/images"):
                os.makedirs("outputs/"+date+"/images")
        self.receive_file("outputs/"+date+"/data.npy")
        while True:
            try:
                data = self.sock.recv(4)
                if not data:
                    break  # Connection closed
                # Unpack the received data
                image_index, = struct.unpack('!I', data)
                self.receive_file("outputs/"+date+"/images/"+str(image_index)+".jpg")
            except Exception as e:
                print("something went wrong", e)
                break

    def run(self):
        with self.running_lock:
            self.running = True
        try:
            if MOCK_GAMEPAD:
                while self.running_lock.acquire() and self.running:
                    self.running_lock.release()
                    self.send_event(1, random.random() - 0.5)
                    time.sleep(0.1)
                self.running_lock.release()
            else:
                while True:
                    for event in pygame.event.get():
                        with self.running_lock:
                            if not self.running:
                                break

                        if event.type == pygame.QUIT:
                            raise Exception("QUIT")
                        elif event.type == pygame.JOYAXISMOTION:
                            self.send_event(event.axis, event.value)
                        # Add a small delay to reduce CPU usage
                        time.sleep(0.01)
        except:
            print("Failed to send event")
            with self.running_lock:
                self.running = False
            self.sock.sendall(b"STOP_")
            self.wait_for_data()
            
    def stop(self):
        with self.running_lock:
            self.running = False
        pygame.quit()
        self.sock.close()

if __name__ == "__main__":

    sender = GamepadSender(SERVER_ADDRESS, SERVER_PORT)
    
    if sender.connect():
        try:
            sender.run()
        except KeyboardInterrupt:
            print("\nStopping gamepad sender...")
        finally:
            sender.stop()
            print("Gamepad sender stopped.")
    else:
        print("Failed to start gamepad sender.")