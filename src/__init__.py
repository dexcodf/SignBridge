"""Sign Language -> Text -> Speech package.

Modules
-------
config            : shared paths and constants
data_loader       : load / clean the Sign Language MNIST dataset
train_model       : train and save the CNN gesture classifier
hand_detector     : OpenCV + MediaPipe hand localisation and pre-processing
gesture_recognizer: load the trained model and turn a frame into a letter
tts_engine        : convert text to speech with Supertonic (Hugging Face)
realtime_demo     : standalone OpenCV webcam loop
"""

__version__ = "0.1.0"
