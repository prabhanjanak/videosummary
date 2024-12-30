import cv2
from transformers import (
    AutoProcessor, AutoModelForImageTextToText,
    PegasusTokenizer, PegasusForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM,
)
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from pydub import AudioSegment
from google.colab import files
from transformers import pipeline

# Load BLIP model for image-to-text transcription
print("[INFO] Loading BLIP model...")
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-image-captioning-large")

# Load Pegasus model for summarization
print("[INFO] Loading Pegasus model...")
pegasus_model_name = "google/pegasus-large"
pegasus_tokenizer = PegasusTokenizer.from_pretrained(pegasus_model_name)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(pegasus_model_name)

# Load Flan-T5 model for summarization
print("[INFO] Loading Flan-T5 model...")
flan_t5_model_name = "google/flan-t5-large"
flan_t5_tokenizer = AutoTokenizer.from_pretrained(flan_t5_model_name)
flan_t5_model = AutoModelForSeq2SeqLM.from_pretrained(flan_t5_model_name)

# Load FalconsAI Summarization Model
print("[INFO] Loading FalconsAI Summarization Model...")
generation_tokenizer = AutoTokenizer.from_pretrained("Falconsai/text_summarization")
generation_model = AutoModelForSeq2SeqLM.from_pretrained("Falconsai/text_summarization")

# Function to process video and extract frames
def process_video(video_path, frame_skip=5):
    print(f"[INFO] Processing video: {video_path}")
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
    print(f"[INFO] Extracted {len(frames)} frames from the video.")
    return frames

# Function to extract audio from video
def extract_audio(video_path):
    print("[INFO] Extracting audio from video...")
    video_clip = VideoFileClip(video_path)
    audio_path = "extracted_audio.wav"
    video_clip.audio.write_audiofile(audio_path)
    return audio_path

# Function to transcribe audio using SpeechRecognition
def transcribe_audio(audio_path):
    print("[INFO] Transcribing audio...")
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_file(audio_path)
    audio.export("temp_audio.wav", format="wav")
    with sr.AudioFile("temp_audio.wav") as source:
        audio_data = recognizer.record(source)
        try:
            transcription = recognizer.recognize_google(audio_data)
            print("[INFO] Audio transcription successful.")
        except sr.UnknownValueError:
            transcription = "[ERROR] Unable to understand audio."
        except sr.RequestError as e:
            transcription = f"[ERROR] API unavailable: {e}"
    return transcription

# Function to transcribe a single frame using BLIP
def transcribe_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=frame_rgb, return_tensors="pt")
    output = blip_model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# Function to generate summary using Pegasus
def summarize_with_pegasus(text, max_length=150, min_length=40):
    print("[INFO] Generating summary using Pegasus...")
    inputs = pegasus_tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
    summary_ids = pegasus_model.generate(inputs.input_ids, max_length=max_length, min_length=min_length, num_beams=5)
    return pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Function to generate summary using Flan-T5
def summarize_with_flan_t5(text, max_length=150, min_length=40):
    print("[INFO] Generating summary using Flan-T5...")
    inputs = flan_t5_tokenizer(text, return_tensors="pt", truncation=True, padding="longest")
    summary_ids = flan_t5_model.generate(inputs.input_ids, max_length=max_length, min_length=min_length, num_beams=5)
    return flan_t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Function to generate summary using FalconsAI
def summarize_with_falconsai(text):
    print("[INFO] Generating summary using FalconsAI...")
    generation_pipe = pipeline("summarization", model=generation_model, tokenizer=generation_tokenizer)
    generated_text = generation_pipe(text, max_length=1024, min_length=100, do_sample=False)[0]['summary_text']
    return generated_text

# Upload video file
print("[INFO] Upload your video file...")
uploaded = files.upload()

# Process each uploaded video
for file_name in uploaded.keys():
    video_path = file_name

    # Step 1: Process video into frames
    frames = process_video(video_path)

    # Step 2: Transcribe each frame
    print("[INFO] Transcribing frames...")
    frame_transcriptions = [transcribe_frame(frame) for frame in frames]

    # Step 3: Extract and transcribe audio
    audio_path = extract_audio(video_path)
    audio_transcription = transcribe_audio(audio_path)

    # Step 4: Combine text from frames and audio
    combined_text = " ".join(frame_transcriptions) + " " + audio_transcription
    print("[INFO] Combined transcription:\n", combined_text[:500], "...")  # Preview first 500 characters

    # Step 5: Generate the summary using one of the models (Uncomment your preferred choice)
    # Using Pegasus
    video_summary_pegasus = summarize_with_pegasus(combined_text)
    # Using Flan-T5
    # video_summary_flan_t5 = summarize_with_flan_t5(combined_text)
    # Using FalconsAI
    video_summary_falconsai = summarize_with_falconsai(combined_text)

    # Step 6: Display the video summary
    print("\n#### Video Summary with Pegasus ####")
    print(video_summary_pegasus)

    print("\n#### Video Summary with FalconsAI ####")
    print(video_summary_falconsai)

    # Uncomment to see Flan-T5 result
    # print("\n#### Video Summary with Flan-T5 ####")
    # print(video_summary_flan_t5)
