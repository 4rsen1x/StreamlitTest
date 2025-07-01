# app.py
import streamlit as st
import tempfile
import soundfile as sf
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import os
from huggingface_hub import login

# For live streaming
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, RTCConfiguration
from aiortc import RTCIceServer  # Only import RTCIceServer from aiortc
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
        self.processor, self.model, self.device = load_model()
        self.text_placeholder = st.empty()
        self.partial_text = ""

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
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
            self.text_placeholder.markdown(f"**Live:** {self.partial_text}")
            # Keep half-overlap
            self.buffer = self.buffer[-(self.sample_rate * self.interval // 2) :]
        return frame


# Configure ICEServers properly
rtc_config = RTCConfiguration(iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])])

webrtc_streamer(
    key="live-transcription",
    mode="SENDRECV",
    audio_processor_factory=WhisperLiveProcessor,
    rtc_configuration=rtc_config,
    media_stream_constraints={"audio": True, "video": False},
)

# Footer
st.markdown("---")
st.caption("Powered by Whisper & Streamlit – live streaming via streamlit-webrtc")
st.markdown("---")
st.caption("Authenticated via Hugging Face token stored in Streamlit secrets")
