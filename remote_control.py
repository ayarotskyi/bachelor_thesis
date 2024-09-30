import socket
import sys
import struct

# Change it to the port of the JetBot (it should be connected to the same network as your computer)
HOST='192.168.0.134'
PORT=8090

def server():
    from Adafruit_MotorHAT import Adafruit_MotorHAT
    from motor import Motor
    from jetbot import JetBot
    i2c_bus = 1
    left_channel = 1
    right_channel = 2
    driver = Adafruit_MotorHAT(i2c_bus=i2c_bus)
    
    left_motor = Motor(driver, left_channel)
    right_motor = Motor(driver, right_channel)
    
    jetbot = JetBot(left_motor, right_motor, save_recording=False)

    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    print('Socket created')
    
    s.bind((HOST,PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')
    
    conn,addr=s.accept()
    print('Connection accepted')

    while True:
        data = conn.recv(4)
        if len(data) != 4:
            raise ConnectionError("Connection closed before receiving complete data")
        char = struct.unpack('!I', data)[0]
        if char == ord('w'):
            jetbot.forward()
        elif char == ord('a'):
            jetbot.left()
        elif char == ord('s'):
            jetbot.backward()
        elif char == ord('d'):
            jetbot.right()
        elif char == ord('e'):
#                Winkel sollte zwischen -1 und 1 sein (simuliert output des Neuronalen Netzes)
            angle = -0.5
            jetbot.turn_by_output(angle)
        else:
            jetbot.stop()

def client():
    import curses
    screen = curses.initscr()
    curses.noecho()
    curses.cbreak()
    screen.keypad(True)
    clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    clientsocket.connect((HOST,PORT))
    while True:
        char = screen.getch()
        data = struct.pack('!I', char)
        clientsocket.sendall(data)
        

if __name__ == "__main__":
    if sys.argv[len(sys.argv) - 1] == 'client':
        client()
    else:
        server()