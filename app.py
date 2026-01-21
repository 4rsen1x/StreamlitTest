# app.py
import streamlit as st
import tempfile
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import os
from huggingface_hub import login
import numpy as np
import time
from audio_recorder_streamlit import audio_recorder
import librosa

# Configuration
LANGUAGE = "ar"  # Arabic
TASK = "transcribe"  # or "translate"


# Load model using transformers
@st.cache_resource(show_spinner=False)
def load_model():
    token = st.secrets.get("hf_token")
    login(token=token)
    model_id = "hifzyml/whisper-quran-model-finetuned_v2"

    # Load model and processor
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)

    # Use CPU (Streamlit Cloud doesn't have GPU)
    device = "cpu"
    model.to(device)

    return model, processor, device


def transcribe_audio(audio_path: str, model, processor, device):
    """Function to transcribe audio file using transformers"""
    try:
        t0 = time.perf_counter()

        # Load audio file
        audio, sr = librosa.load(audio_path, sr=16000)

        # Process audio
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        inputs = inputs.to(device)

        # Generate transcription
        with torch.no_grad():
            forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)
            predicted_ids = model.generate(inputs.input_features, forced_decoder_ids=forced_decoder_ids)

        # Decode
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        elapsed = time.perf_counter() - t0
        return transcription.strip(), elapsed
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return "", 0


# Global model loading
model, processor, device = load_model()

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

        # Transcribe
        try:
            with st.spinner("Transcribing..."):
                transcription, elapsed = transcribe_audio(tmp_file.name, model, processor, device)

            if transcription:
                st.success("**Transcription:**")
                st.write(transcription)
                st.info(f"⏱️ Transcription took {elapsed:.2f} seconds")
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
            # Transcribe
            with st.spinner("Transcribing uploaded file..."):
                transcription, elapsed = transcribe_audio(tmp_file.name, model, processor, device)

            if transcription:
                st.success("**Transcription:**")
                st.write(transcription)
                st.info(f"⏱️ Transcription took {elapsed:.2f} seconds")

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
