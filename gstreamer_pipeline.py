"""
The gstreamer pipeline is necessary to instantiate an opencv VideoCapture. These Settings are perfectly fit for the Raspberry Pi Camera Module 8MP v2.
If you're using a different camera make sure you find the best settings. Especially the sensor mode is important.
"""
def gstreamer_pipeline(
    sensor_mode=3,
    capture_width=816,
    capture_height=616,
    display_width=816,
    display_height=616,
    framerate=30,
    saturation=1.0,
    brightness=0.0,
    hue=0.0,
    contrast=1.0,
#     flip_method=2,
):
    return (
        "nvarguscamerasrc sensor-mode=%d ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv ! "
        "videobalance saturation=%.1f brightness=%.1f hue=%.1f contrast=%.1f ! "
#         "nvvidconv flip-method=%d ! "
        "nvvidconv ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "appsink"
#         "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_mode,
            capture_width,
            capture_height,
            framerate,
            saturation,
            brightness,
            hue,
            contrast,
#             flip_method,
            display_width,
            display_height,
        )
    )