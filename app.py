import streamlit as st
import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import tempfile
import os

# Load YOLOv8 model (assuming ultralytics library is used)
yolo_model = torch.hub.load('ultralytics/yolov8', 'yolov8')

# Load Florence model from Microsoft
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize a summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to process video and get frames
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 5 == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

# Function to analyze frames
def analyze_frames(frames):
    transcriptions = []
    for frame in frames:
        # YOLOv8 detection (This part is optional, as we focus on visual understanding)
        results_yolo = yolo_model(frame)
        detections = results_yolo.pandas().xyxy[0]

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Florence description
        inputs = blip_processor(images=frame_rgb, return_tensors="pt")
        out = blip_model.generate(**inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)

        transcriptions.append(caption)

    # Combine transcriptions into a single text
    combined_transcription = " ".join(transcriptions)
    return combined_transcription

# Streamlit app
st.title("Video to Text Summary Tool")
st.write("Upload a video file to analyze its content and generate a summary.")

# Upload video file
uploaded_file = st.file_uploader("Upload your video file", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    # Save the uploaded video file
    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video_file.write(uploaded_file.read())
    video_path = temp_video_file.name

    # Display video
    st.video(video_path)

    # Process video to get frames
    st.write("### Step 1: Processing Video")
    with st.spinner("Processing video and extracting frames..."):
        frames = process_video(video_path)
    st.success("Frames extracted successfully!")

    # Analyze frames
    st.write("### Step 2: Analyzing Frames")
    with st.spinner("Analyzing frames..."):
        combined_transcription = analyze_frames(frames)
    st.success("Frames analyzed successfully!")

    # Generate video summary
    st.write("### Step 3: Generating Summary")
    with st.spinner("Summarizing the video..."):
        summary_result = summarizer(combined_transcription, max_length=150, min_length=40, do_sample=False)
        video_summary = summary_result[0]['summary_text']

    st.write("#### Video Summary:")
    st.text_area("Summary", video_summary, height=150)

    # Cleanup temporary file
    os.remove(video_path)
    st.success("Process completed successfully!")
