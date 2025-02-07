import os
from enum import Enum

ModelVersion = Enum('ModelVersion', [('TEST', 0), ('LSTM', 1), ('LARQ', 2)])

def load_model(model_path: str, model_version: ModelVersion = ModelVersion.LSTM):
    try:
        from keras import Sequential
        from keras.layers import Flatten, Dense, Lambda, ConvLSTM2D, BatchNormalization, TimeDistributed, Conv2D
    except:
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Flatten, Dense, Lambda, ConvLSTM2D, BatchNormalization, TimeDistributed, Conv2D
    from larq.layers import QuantConv3D
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    try:
        if model_version == ModelVersion.TEST:
            model = Sequential()
            model.add(Lambda(lambda x: (x/255), input_shape = (400, 400, 1)))
            model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu', name='conv1'))
            model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu', name='conv2'))
            model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu', name='conv3'))
            model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='conv4'))
            model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='conv5')) 
            model.add(Flatten())
            model.add(Dense(100))
            model.add(Dense(50))
            model.add(Dense(10))
            model.add(Dense(2))
        elif model_version == ModelVersion.LSTM:
            model = Sequential([
                Lambda(lambda x: x/255, input_shape=(4, 100, 400, 1)),
                ConvLSTM2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu', 
                           data_format='channels_last', name='conv_lstm1', return_sequences=True),
                TimeDistributed(BatchNormalization()),
                ConvLSTM2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu', 
                           data_format='channels_last', name='conv_lstm2', return_sequences=True),
                TimeDistributed(BatchNormalization()),
                ConvLSTM2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu', 
                           data_format='channels_last', name='conv_lstm3', return_sequences=True),
                ConvLSTM2D(64, kernel_size=(3, 3), activation='relu', 
                           data_format='channels_last', name='conv_lstm4', return_sequences=True),
                ConvLSTM2D(64, kernel_size=(3, 3), activation='relu', 
                           data_format='channels_last', name='conv_lstm5'),
                Flatten(),
                Dense(100),
                Dense(50),
                Dense(10),
                Dense(2, activation='tanh')
            ])
        elif model_version == ModelVersion.LARQ:
            model = Sequential([
                Lambda(lambda x: x / 255, input_shape=(4, 100, 400, 1)),

                QuantConv3D(24, kernel_size=(3, 5, 5), strides=(1, 2, 2),
                            activation='relu', name='quant_conv3d_1', use_bias=False, padding="same"),
                BatchNormalization(),

                QuantConv3D(36, kernel_size=(3, 5, 5), strides=(1, 2, 2),
                            activation='relu', name='quant_conv3d_2', use_bias=False, padding="same"),
                BatchNormalization(),

                QuantConv3D(48, kernel_size=(3, 5, 5), strides=(1, 2, 2),
                            activation='relu', name='quant_conv3d_3', use_bias=False, padding="same"),
                BatchNormalization(),

                QuantConv3D(64, kernel_size=(3, 3, 3), activation='relu',
                            name='quant_conv3d_4', use_bias=False, padding="same"),

                QuantConv3D(64, kernel_size=(3, 3, 3), activation='relu',
                            name='quant_conv3d_5', use_bias=False, padding="same"),

                Flatten(),
                Dense(100),
                Dense(50),
                Dense(10),
                Dense(2, activation='tanh')
            ])

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