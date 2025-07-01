# app.py
import streamlit as st
import tempfile
import soundfile as sf
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import os
from huggingface_hub import login

# For live streaming
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import numpy as np


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


# Global model loading
processor, model, device = load_model()

# Page setup
st.set_page_config(page_title="🎙️ Audio Transcriber", layout="centered")
st.title("🎙️ Audio Transcriber (Dark Theme)")
st.markdown("_Record or upload a WAV, then transcribe in seconds._")

# Section: Live Transcription
st.markdown("---")
st.subheader("1. Live Transcription")


class WhisperLiveProcessor(AudioProcessorBase):
    def __init__(self):
        self.buffer = np.zeros((0,), dtype=np.float32)
        self.interval = 2  # seconds
        self.sample_rate = 16000
        # Use global variables instead of reloading
        self.processor = processor
        self.model = model
        self.device = device
        self.partial_text = ""

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        try:
            # Convert to mono float32
            data = frame.to_ndarray()
            if data.ndim > 1:
                audio = data.mean(axis=0)
            else:
                audio = data
            audio = audio.astype(np.float32) / 32768.0
            self.buffer = np.concatenate([self.buffer, audio])

            # Transcribe when enough audio collected
            if len(self.buffer) >= self.sample_rate * self.interval:
                segment = self.buffer[-self.sample_rate * self.interval :]
                inputs = self.processor(segment, sampling_rate=self.sample_rate, return_tensors="pt").input_features.to(
                    self.device
                )
                generated_ids = self.model.generate(inputs, max_length=448)
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                self.partial_text += " " + text

                # Update UI in session state instead of direct placeholder
                if "live_transcription" not in st.session_state:
                    st.session_state.live_transcription = ""
                st.session_state.live_transcription = self.partial_text

                # Keep half-overlap
                self.buffer = self.buffer[-(self.sample_rate * self.interval // 2) :]
        except Exception as e:
            st.error(f"Error in audio processing: {e}")

        return frame


# Display live transcription
if "live_transcription" in st.session_state:
    st.markdown(f"**Live Transcription:** {st.session_state.live_transcription}")

# Simple webrtc_streamer without RTCConfiguration to avoid conflicts
ctx = webrtc_streamer(
    key="live-transcription",
    mode="SENDRECV",
    audio_processor_factory=WhisperLiveProcessor,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

# Clear transcription button
if st.button("Clear Transcription"):
    if "live_transcription" in st.session_state:
        st.session_state.live_transcription = ""
        st.rerun()

# Show connection status
if ctx.state.playing:
    st.success("🔴 Recording... Speak into your microphone")
else:
    st.info("👆 Click 'Start' to begin live transcription")

# Footer
st.markdown("---")
st.caption("Powered by Whisper & Streamlit – live streaming via streamlit-webrtc")
st.markdown("---")
st.caption("Authenticated via Hugging Face token stored in Streamlit secrets")
