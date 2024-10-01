from Adafruit_MotorHAT import Adafruit_MotorHAT
from motor import Motor
from camera import Camera
import atexit
import math
import time


"""
The JetBot class is used to create an instance of the JetBot allowing to control the motors and the camera via provided functions.
The minimum speed for the motors to start driving is a little less then 0.3.
"""
class JetBot():
    
    left_motor = None
    right_motor = None
    
    def __init__(self, left_motor, right_motor, save_recording=True, *args, **kwargs):
        self.left_motor = left_motor
        self.right_motor = right_motor
        self.camera = None
        
        atexit.register(self.stop)
        
#   Set different speeds for both motors.
    def set_motors(self, left_speed, right_speed):
        self.left_motor.setSpeed(left_speed)
        self.right_motor.setSpeed(right_speed)

#   Set same speed for both motors (positive)
    def forward(self, speed=0.3):
        print('forward, ', speed)
        self.left_motor.setSpeed(speed)
        self.right_motor.setSpeed(speed)

#   Set same speed for both motors (negative)
    def backward(self, speed=0.3):
        print('backward, ', speed)
        self.left_motor.setSpeed(-speed)
        self.right_motor.setSpeed(-speed)

#   Turn left. Wheels turn in opposite direction at same speed.
    def left(self, speed=0.3):
        print('left, ', speed)
        self.left_motor.setSpeed(-speed)
        self.right_motor.setSpeed(speed)

#   Turn right. Wheels turn in opposite direction at same speed.
    def right(self, speed=0.3):
        print('right, ', speed)
        self.left_motor.setSpeed(speed)
        self.right_motor.setSpeed(-speed)
                

#   Stop the wheels from turning.
    def stop(self):
        print('stop')
        self.left_motor.setSpeed(0)
        self.right_motor.setSpeed(0)
        
#   Berechnet aus bekannter Winkelgeschwindigkeit die Zeit, die gedreht werden muss, um uebergebenen Winkel zu drehen.
#   Hier bleibt Auto zum Drehen stehen. Soll es dennoch fahren muss die Geschwindigkeitsdifferenz und die dazugehoerige Winkelgeschwindigkeit angepasst werden.
    def turn_by_output(self, steering_output = 0):
        print('steer')
        if steering_output != 0:
#             current_speed_left = self.left_motor.speed
#             current_speed_right = self.right_motor.speed
            
#             fuer Testzwecke, bei test speed dreht sich der JetBot aus dem Stand
            test_speed = 0
            speed_dif = 1 - abs(test_speed)
            speed_dif = 0.5
            
#             if steering_output > 0:
#                 speed_dif = 1 - abs(current_speed_left)
#             elif steering_output < 0:
#                 speed_dif = 1 - abs(current_speed_right)

#             Falls Geschwindigkeitsdifferenz nicht fest waere. Dann muesste aber Winkelgeschwindigkeit ermittelt werden. Eine Liste mit einigen gemessen Werten ist in der BA zu finden.
            angular_velocity = 2.55
    
            angle = abs(steering_output * 180)
            print('angle: ', angle)
            angle_rad = (angle * math.pi) / 180
            needed_time = angle_rad/angular_velocity
            print('time: ', needed_time)
            time_goal = time.time() + needed_time
            while time.time() < time_goal:
                if steering_output < 0:
#                     Bei echter Lenkung weglassen, damit jetziges Tempo beigehalten wird
                    self.left_motor.setSpeed(test_speed)
                    self.right_motor.setSpeed(speed_dif)
                else:
#                     Bei echter Lenkung weglassen, damit jetziges Tempo beigehalten wird
                    self.right_motor.setSpeed(test_speed)
                    self.left_motor.setSpeed(speed_dif)
            self.stop()
        
#   Nimmt ein Video auf. Der Parameter Highlighting bestimmt, ob auf der Aufnahme bereits Rechtecke eingezeichnet werden sollen
    def record_video(self, highlighting=False):
        if highlighting:
            self.camera.capture_highlighted_video()
        else:
            self.camera.capture_video()

#   Nimmt ein einzelnes Bild auf
    def take_picture(self):
        self.camera.take_picture()
    
    def stop_camera(self):
        self.camera.stop()