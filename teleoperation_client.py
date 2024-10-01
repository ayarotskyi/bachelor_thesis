import pygame
import socket
import struct
import time

class GamepadSender:
    def __init__(self, server_address, server_port):
        self.server_address = server_address
        self.server_port = server_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.running = False

        # Initialize Pygame
        pygame.init()
        pygame.joystick.init()

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
        # - B (unsigned char) for event type (1 byte, using 0 for axis events)
        # - B (unsigned char) for axis number (1 byte)
        # - f (float) for axis value (4 bytes)
        packed_data = struct.pack('!BBf', 0, axis, value)
        
        try:
            self.sock.sendall(packed_data)
            print(f"Sent axis event: Axis {axis}, Value {value:.2f}")
        except Exception as e:
            print(f"Failed to send event: {e}")

    def run(self):
        self.running = True
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.JOYAXISMOTION:
                    self.send_event(event.axis, event.value)

            # Add a small delay to reduce CPU usage
            time.sleep(0.01)

    def stop(self):
        self.running = False
        pygame.quit()
        self.sock.close()

if __name__ == "__main__":
    SERVER_ADDRESS = "192.168.0.134"  # Change this to your server's address
    SERVER_PORT = 8090  # Change this to your server's port

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