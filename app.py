# app.py
import streamlit as st
import tempfile
import soundfile as sf
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import os
from huggingface_hub import login
import numpy as np
import time
from audio_recorder_streamlit import audio_recorder

st.markdown(
    """
    <style>
    /* Use a clean Arabic-compatible font and set RTL direction */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Tahoma', 'Arial', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Load model and processor from Hugging Face
@st.cache_resource(show_spinner=False)
def load_model():
    token = st.secrets.get("hf_token")
    login(token=token)
    model_id = "hifzyml/whisper-arabic-quran-finetuned"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device


def transcribe_audio(audio_data, processor, model, device):
    """Function to transcribe audio data"""
    try:
        # Process audio
        inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        # Generate transcription
        with torch.no_grad():
            generated_ids = model.generate(input_features, max_length=448)
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return transcription.strip()
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return ""


# Global model loading
processor, model, device = load_model()

# Page setup
st.set_page_config(page_title="🎙️ Audio Transcriber", layout="centered")
st.title("🎙️ Audio Transcriber")
st.markdown("_Record audio and get instant transcription._")

# Method 1: Audio Recorder Streamlit (Recommended)
st.markdown("---")
st.subheader("🎤 Method 1: Quick Audio Recording")
st.markdown("Click the microphone button below to record audio:")

# Record audio
audio_bytes = audio_recorder(
    text="Click to record",
    recording_color="#e74c3c",
    neutral_color="#34495e",
    icon_name="microphone",
    icon_size="2x",
    pause_threshold=2.0,
    sample_rate=16000,
)

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

    # Convert bytes to numpy array for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file.flush()

        # Load and process audio
        try:
            audio_data, sample_rate = sf.read(tmp_file.name)

            # Resample if needed
            if sample_rate != 16000:
                from scipy import signal

                audio_data = signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate))

            # Ensure mono
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            # Transcribe
            with st.spinner("Transcribing..."):
                transcription = transcribe_audio(audio_data, processor, model, device)

            if transcription:
                st.success("**Transcription:**")
                st.write(transcription)
            else:
                st.warning("No speech detected or transcription failed.")

        except Exception as e:
            st.error(f"Error processing audio: {e}")
        finally:
            # Clean up temp file
            os.unlink(tmp_file.name)

# Method 2: File Upload
st.markdown("---")
st.subheader("📁 Method 2: Upload Audio File")
uploaded_file = st.file_uploader(
    "Choose an audio file", type=["wav", "mp3", "m4a", "flac", "ogg"], help="Upload a WAV, MP3, M4A, FLAC, or OGG file"
)

if uploaded_file is not None:
    # Display audio player
    st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file.flush()

        try:
            # Load and process audio
            audio_data, sample_rate = sf.read(tmp_file.name)

            # Resample if needed
            if sample_rate != 16000:
                from scipy import signal

                audio_data = signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate))

            # Ensure mono
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            # Transcribe
            with st.spinner("Transcribing uploaded file..."):
                transcription = transcribe_audio(audio_data, processor, model, device)

            if transcription:
                st.success("**Transcription:**")
                st.write(transcription)

                # Option to download transcription
                st.download_button(
                    label="Download Transcription", data=transcription, file_name="transcription.txt", mime="text/plain"
                )
            else:
                st.warning("No speech detected or transcription failed.")

        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
        finally:
            # Clean up temp file
            os.unlink(tmp_file.name)

# Instructions
st.markdown("---")
st.subheader("📋 Instructions")
st.markdown("""
**Method 1 (Recommended):** Use the microphone button for quick recordings
- Click and hold to record
- Release to stop and transcribe
- Works great for short voice notes

**Method 2:** Upload pre-recorded audio files
- Supports multiple formats
- Good for longer recordings
- Can download transcriptions
""")

# Footer
st.markdown("---")
st.caption("🤖 Powered by Whisper (Arabic Quran Fine-tuned) & Streamlit")
st.caption("🔐 Authenticated via Hugging Face token")
