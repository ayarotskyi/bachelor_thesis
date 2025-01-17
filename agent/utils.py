import os

def load_model(model_path: str):
    try:
        import keras
    except:
        import tensorflow.keras as keras
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    try:
        model = keras.Sequential()
        model.add(keras.layers.Lambda(lambda x: (x/255), input_shape = (400, 400, 1)))
        model.add(keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu', name='conv1'))
        model.add(keras.layers.Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu', name='conv2'))
        model.add(keras.layers.Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu', name='conv3'))
        model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='conv4'))
        model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='conv5')) 
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(100))
        model.add(keras.layers.Dense(50))
        model.add(keras.layers.Dense(10))
        model.add(keras.layers.Dense(2))
        
        model.load_weights(model_path)
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")
    
def init_jetbot():
    try:
        i2c_bus = 1
        left_channel = 1
        right_channel = 2
        from motor import Motor
        from jetbot import JetBot
        from Adafruit_MotorHAT import Adafruit_MotorHAT
        driver = Adafruit_MotorHAT(i2c_bus=i2c_bus)
        left_motor = Motor(driver, left_channel)
        right_motor = Motor(driver, right_channel)
        jetbot = JetBot(left_motor, right_motor, save_recording=False)
    except:
        jetbot = None
        
    return jetbot

def calculate_motor_speeds(x, y):
    rotation_quotient = 0.5
    # Calculate left and right motor powers
    left_power = -y + x * rotation_quotient
    right_power = -y - x * rotation_quotient

    # Normalize powers to ensure they're within -1 to 1 range
    max_power = max(abs(left_power), abs(right_power), 1)
    left_power /= max_power
    right_power /= max_power
    
    return left_power, right_power