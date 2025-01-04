import atexit
from Adafruit_MotorHAT import Adafruit_MotorHAT

"""
The Motor class is used to create an instance of the Motor which controls one wheel of the JetBot.
So to control both wheels you need to create two instances with the corresponding channels for each wheel.
"""
class Motor():
    # config
    alpha = 1
    beta = 0

    def __init__(self, driver, channel, *args, **kwargs):
        self.speed=0
        self._driver = driver
        self._motor = self._driver.getMotor(channel)
        if(channel == 1):
            self._ina = 1
            self._inb = 0
        else:
            self._ina = 2
            self._inb = 3
        atexit.register(self._release)
        
    # Sets the wheel that's connected to the motor. The value has to be in range [-1, 1]
    def setSpeed(self, value):
        self.speed=value
        mapped_value = int(255 * (self.alpha * value + self.beta))
        speed = min(max(abs(mapped_value), 0), 255)
        self._motor.setSpeed(speed)
        if mapped_value < 0:
            self._motor.run(Adafruit_MotorHAT.FORWARD)
#             self._driver._pwm.setPWM(self._ina,0,0)
#             self._driver._pwm.setPWM(self._inb,0,speed*16)
        else:
            self._motor.run(Adafruit_MotorHAT.BACKWARD)
#             self._driver._pwm.setPWM(self._ina,0,speed*16)
#             self._driver._pwm.setPWM(self._inb,0,0)

    def _release(self):
        """Stops motor by releasing control"""
        self._motor.run(Adafruit_MotorHAT.RELEASE)
#         self._driver._pwm.setPWM(self._ina,0,0)
#         self._driver._pwm.setPWM(self._inb,0,0)