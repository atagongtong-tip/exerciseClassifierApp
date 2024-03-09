import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import cv2
from collections import deque
import os
import subprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras

# Specify the height and width to which each video frame will be resized in our dataset.
IMG_SIZE = 224
SEQUENCE_LENGTH = 30
NUM_FEATURES = 2048

# loading the saved model
loaded_model = load_model("C:\\Users\\Arvin\\Desktop\\exerciseClassifierApp\\80.h5")

# Specify the list containing the names of the classes used for training.
CLASSES_LIST = ["Barbel_Biceps_Curl", "Barbell_Row", "Dead_Lift", "Jump_and_Jacks", "Lateral_Raise", "Lunges", "PushUp", "Squat", "Deadlift"]

# Create an instance of the ImageDataGenerator for data augmentation
data_generator = ImageDataGenerator(
    horizontal_flip=True,
    brightness_range=(0.7, 1.3),
)

# Function to process video files
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",  # Corrected line
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

# Load the feature extractor globally
feature_extractor = build_feature_extractor()

# Prepare all videos
def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values

    frame_masks = np.zeros(shape=(num_samples, SEQUENCE_LENGTH, NUM_FEATURES), dtype="bool")
    frame_features = np.zeros(shape=(num_samples, SEQUENCE_LENGTH, NUM_FEATURES), dtype="float32")

    for idx, path in enumerate(video_paths):
        frames = load_video(os.path.join(root_dir, path))
        frames = frames[None, ...]

        temp_frame_mask = np.zeros(shape=(1, SEQUENCE_LENGTH, NUM_FEATURES), dtype="bool")
        temp_frame_features = np.zeros(shape=(1, SEQUENCE_LENGTH, NUM_FEATURES), dtype="float32")

        video_length = frames.shape[1]

        # Temporal Padding
        if video_length < SEQUENCE_LENGTH:
            temp_frame_mask[:, video_length:, :] = 0
            temp_frame_features[:, video_length:, :] = 0

        # Temporal Sampling
        if video_length > SEQUENCE_LENGTH:
            frame_indices = np.linspace(0, video_length - 1, SEQUENCE_LENGTH, dtype=int)
            frames = frames[:, frame_indices]

        # Data Augmentation
        for i, frame in enumerate(frames[0]):
            frame = data_generator.random_transform(frame)
            temp_frame_features[0, i, :] = feature_extractor.predict(frame[None, ...])
            temp_frame_mask[0, i, :] = 1  # 1 = not masked, 0 = masked

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels


# Predict on video
def predict_on_video(video_file_path, output_file_path):
    video_reader = cv2.VideoCapture(video_file_path)
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 
                                   video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)

    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break
        resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        normalized_frame = resized_frame / 255
        frame_features = feature_extractor.predict(normalized_frame[np.newaxis, ...])

        frames_queue.append(frame_features)

        if len(frames_queue) == SEQUENCE_LENGTH:
            # Convert frames_queue to a numpy array and add a batch dimension
            frames_array = np.array(frames_queue)
            frames_array = np.expand_dims(frames_array, axis=0)  # Add batch dimension

            # Create frame masks (all ones) with the same shape as frames_array
            frame_masks = np.ones((frames_array.shape[0], SEQUENCE_LENGTH, NUM_FEATURES))

            if frame_masks.shape != (None, SEQUENCE_LENGTH, NUM_FEATURES):
                print("Error: Frame masks do not have the correct shape.")
                return

            # Predict using both frames_array and frame_masks
            predicted_labels_probabilities = loaded_model.predict([frames_array, frame_masks])[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = CLASSES_LIST[predicted_label]

            # Draw predicted class name on the frame
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write frame with predicted class name to output video
        video_writer.write(frame)

    video_reader.release()
    video_writer.release()





# Main Streamlit application
def main():  
    st.title('Exercise Classifier App')
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mpeg"])
    if uploaded_file is not None:
        with open(os.path.join("C:\\Users\\Arvin\\Desktop\\exerciseClassifierApp\\vidupload\\", uploaded_file.name.split("/")[-1]), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File Uploaded Successfully")
                       
        if st.button('Classify The Video'):
            output_video_file_path = "C:\\Users\\Arvin\\Desktop\\exerciseClassifierApp\\vidoutput\\" + uploaded_file.name.split("/")[-1].split(".")[0] + "_output1.mp4"
            with st.spinner('Wait for it...'):
                predict_on_video("C:\\Users\\Arvin\\Desktop\\exerciseClassifierApp\\vidupload\\" + uploaded_file.name.split("/")[-1], output_video_file_path)
                subprocess.call(['ffmpeg', '-y', '-i', output_video_file_path, '-vcodec', 'libx264', '-f', 'mp4', 'C:\\Users\\Arvin\\Desktop\\exerciseClassifierApp\\vidoutput\\output1.mp4'], shell=True)
                st.success('Done!')
            
            st.video("C:\\Users\\Arvin\\Desktop\\exerciseClassifierApp\\vidoutput\\output1.mp4")
    
    else:
        st.text("Please upload a video file")


if __name__ == '__main__':
    main()
