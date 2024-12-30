# Video Summarization and Classification Tool

## Overview

This application allows users to upload a video file, process it frame by frame, generate text summaries for each frame using a vision-language model (BLIP), and create an overall summary using a text generation model (Flan-T5). The frame-wise summaries and overall video summary are displayed interactively on a Streamlit interface.

## Features

1. Upload a video file in formats like MP4, AVI, or MOV.
2. Extract frames from the video at regular intervals.
3. Generate a text summary for each frame using the BLIP model.
4. Correct frame summaries using a BERT-based NLP model.
5. Combine all frame summaries and generate an overall summary using Flan-T5.
6. Display frame-wise summaries and the overall video summary interactively.

## Requirements

- Python 3.7 or higher
- Dependencies listed in `requirements.txt`
