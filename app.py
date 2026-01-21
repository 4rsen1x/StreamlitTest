# app.py
import streamlit as st
import tempfile
import soundfile as sf
from faster_whisper import WhisperModel
import os
from huggingface_hub import login
import numpy as np
import time
from audio_recorder_streamlit import audio_recorder

# st.markdown(
#     """
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@100..900&display=swap');
#     /* Use a clean Arabic-compatible font and set RTL direction */
#     html, body, [data-testid="stAppViewContainer"] {
#         font-family: 'Tahoma', 'Arial', sans-serif;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )
# Configuration for faster-whisper
DEVICE = "cuda"           # "cuda" if you have GPU, else "cpu"
COMPUTE_TYPE = "float16"  # fast on GPU; if CPU, try "int8" or "int8_float16"
LANGUAGE = "ar"           # e.g. "ar" for Arabic, "en" for English, or None to auto-detect
TASK = "transcribe"       # or "translate"
VAD_FILTER = True         # voice activity detection; helps skip long silences
BEAM_SIZE = 5

# Load model using faster-whisper
@st.cache_resource(show_spinner=False)
def load_model():
    token = st.secrets.get("hf_token")
    login(token=token)
    model_id = "hifzyml/whisper-quran-model-finetuned_v2"

    # Load faster-whisper model
    model = WhisperModel(
        model_id,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )
    return model


def transcribe_audio(audio_path: str, model):
    """Function to transcribe audio file using faster-whisper"""
    try:
        t0 = time.perf_counter()
        segments, _ = model.transcribe(
            audio_path,
            language=LANGUAGE,
            task=TASK,
            beam_size=BEAM_SIZE,
            vad_filter=VAD_FILTER,
            temperature=0.0,
            word_timestamps=False,
        )
        final_text = " ".join(seg.text.strip() for seg in segments).strip()
        elapsed = time.perf_counter() - t0
        return final_text, elapsed
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return "", 0


# Global model loading
model = load_model()

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
                transcription, elapsed = transcribe_audio(tmp_file.name, model)

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
                transcription, elapsed = transcribe_audio(tmp_file.name, model)

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
