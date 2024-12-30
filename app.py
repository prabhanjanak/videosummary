import cv2
import os
import spacy
from transformers import (
    AutoProcessor, AutoModelForImageTextToText,
    AutoTokenizer, AutoModelForSeq2SeqLM
)
import streamlit as st

# Streamlit app title
st.title("Video Summarization Tool with Frame-wise Summaries")

# Load models
@st.cache_resource
def load_models():
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-image-captioning-large")
    flan_t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
    flan_t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl")
    bert_nlp = spacy.load("en_core_web_sm")
    return processor, blip_model, flan_t5_tokenizer, flan_t5_model, bert_nlp

processor, blip_model, flan_t5_tokenizer, flan_t5_model, bert_nlp = load_models()

# Process video and extract frames
def process_video(video_path, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

# Transcribe a single frame using BLIP
def transcribe_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=frame_rgb, return_tensors="pt")
    output = blip_model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

# Correct sentence using BERT-based NLP model
def correct_sentence(text):
    doc = bert_nlp(text)
    return " ".join([token.text for token in doc])

# Summarize text using Flan-T5
def summarize_text_with_flan_t5(text, max_length=200, min_length=50):
    inputs = flan_t5_tokenizer(text, return_tensors="pt", truncation=True, padding="longest")
    summary_ids = flan_t5_model.generate(inputs.input_ids, max_length=max_length, min_length=min_length, num_beams=5)
    return flan_t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Upload video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    # Step 1: Process video into frames
    frames = process_video("temp_video.mp4")

    # Step 2: Generate and display frame-wise summaries
    st.write("Processing frames and generating summaries...")
    frame_summaries = []

    for idx, frame in enumerate(frames):
        transcription = transcribe_frame(frame)
        corrected_transcription = correct_sentence(transcription)
        frame_summaries.append(corrected_transcription)
        st.subheader(f"Frame {idx + 1}:")
        st.write(corrected_transcription)

    # Step 3: Combine frame summaries
    combined_text = " ".join(frame_summaries)

    # Step 4: Generate overall summary using Flan-T5
    st.write("Generating overall summary...")
    overall_summary = summarize_text_with_flan_t5(combined_text)

    # Display results
    st.subheader("Overall Video Summary")
    st.write(overall_summary)
