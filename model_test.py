import keras
import json
import cv2
import numpy as np
import pandas as pd
import os

def preprocess_image(image_path):
    image = cv2.resize(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), (400, 200))[100:, :]
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    blurred = cv2.GaussianBlur(image, (15, 15), 10)
    median_intensity = np.median(blurred)
    lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
    upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))
    canny_edges = cv2.Canny(blurred, 
                             threshold1=lower_threshold, 
                             threshold2=upper_threshold, 
                             apertureSize=5)
    return canny_edges

if __name__ == '__main__':
    keras.config.enable_unsafe_deserialization()
    with open('./model_nvid_angle.json', 'r') as json_file:
        model_json = json.load(json_file)
    loaded_model = keras.models.model_from_json(model_json)
    loaded_model.load_weights('./model.weights.h5')

    df_array = pd.read_csv("/Users/andrewyarotskyi/reduced_data/data.csv").to_numpy()
    index = 44
    image_memory_stack = []
    overflows_starting_timestamp = False
    for i in range(0, 4):
        current_index = index - i
        image_filename = f"{current_index}.png"
        image_path = os.path.join("/Users/andrewyarotskyi/reduced_data/images", image_filename)
        if (not overflows_starting_timestamp 
            and current_index > 0 
            and int(df_array[current_index][2]) > 0
            and os.path.exists(image_path)):
            image_memory_stack.append(preprocess_image(image_path))
        else:
            overflows_starting_timestamp = True
            image_memory_stack.append(np.zeros((100, 400)))

    input = np.asarray([image_memory_stack])
    print(loaded_model.predict(input))