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
import io

# Try different audio recording approaches
try:
    from streamlit_mic_recorder import mic_recorder

    MIC_RECORDER_AVAILABLE = True
except ImportError:
    MIC_RECORDER_AVAILABLE = False

try:
    from audio_recorder_streamlit import audio_recorder

    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False


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


def transcribe_audio(audio_data, processor, model, device, sample_rate=16000):
    """Function to transcribe audio data"""
    try:
        # Ensure audio is float32 and normalized
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                audio_data = audio_data.astype(np.float32)

        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample if necessary
        if sample_rate != 16000:
            from scipy import signal

            audio_data = signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate))

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
st.set_page_config(page_title="🎙️ Real-Time Audio Transcriber", layout="centered")
st.title("🎙️ Real-Time Audio Transcriber")
st.markdown("_Multiple approaches for live audio transcription._")

# Initialize session state for live transcription
if "live_transcription" not in st.session_state:
    st.session_state.live_transcription = ""
if "transcription_history" not in st.session_state:
    st.session_state.transcription_history = []
if "is_continuous" not in st.session_state:
    st.session_state.is_continuous = False

# Method 1: streamlit-mic-recorder (Best for real-time)
if MIC_RECORDER_AVAILABLE:
    st.markdown("---")
    st.subheader("🎤 Method 1: Real-Time Mic Recorder (Recommended)")
    st.markdown("This provides the closest to real-time transcription:")

    col1, col2, col3 = st.columns(3)

    with col1:
        continuous_mode = st.checkbox("Continuous Mode", value=st.session_state.is_continuous)
        st.session_state.is_continuous = continuous_mode

    with col2:
        if st.button("Clear All Transcriptions"):
            st.session_state.live_transcription = ""
            st.session_state.transcription_history = []
            st.rerun()

    # Real-time recording with mic_recorder
    audio_data = mic_recorder(
        start_prompt="🎙️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        just_once=not st.session_state.is_continuous,
        use_container_width=True,
        callback=None,
        args=(),
        kwargs={},
        key="mic_recorder",
    )

    if audio_data is not None:
        # Extract audio bytes and sample rate
        audio_bytes = audio_data["bytes"]
        sample_rate = audio_data.get("sample_rate", 44100)

        # Convert bytes to numpy array
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

        # Display audio player
        st.audio(audio_bytes, format="audio/wav")

        # Transcribe
        with st.spinner("Transcribing..."):
            transcription = transcribe_audio(audio_np, processor, model, device, sample_rate)

        if transcription:
            # Update live transcription
            if st.session_state.is_continuous:
                st.session_state.live_transcription += " " + transcription
                st.session_state.transcription_history.append(transcription)
            else:
                st.session_state.live_transcription = transcription

            st.success("**Latest Transcription:**")
            st.write(transcription)

    # Show live/continuous transcription
    if st.session_state.live_transcription:
        st.markdown("### 📝 Live Transcription")
        st.text_area("Full Transcription", st.session_state.live_transcription, height=150, key="live_display")

        # Download option
        st.download_button(
            "💾 Download Transcription",
            st.session_state.live_transcription,
            file_name="live_transcription.txt",
            mime="text/plain",
        )

# Method 2: audio-recorder-streamlit (Fallback)
elif AUDIO_RECORDER_AVAILABLE:
    st.markdown("---")
    st.subheader("🎤 Method 2: Audio Recorder (Fallback)")
    st.info("Install streamlit-mic-recorder for better real-time experience: `pip install streamlit-mic-recorder`")

    audio_bytes = audio_recorder(
        text="Click to record",
        recording_color="#e74c3c",
        neutral_color="#34495e",
        icon_name="microphone",
        icon_size="2x",
        pause_threshold=1.0,
        sample_rate=16000,
    )

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

        # Convert bytes to numpy array
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file.flush()

            try:
                audio_data, sample_rate = sf.read(tmp_file.name)

                with st.spinner("Transcribing..."):
                    transcription = transcribe_audio(audio_data, processor, model, device, sample_rate)

                if transcription:
                    st.success("**Transcription:**")
                    st.write(transcription)

                    # Add to continuous transcription if enabled
                    if st.session_state.is_continuous:
                        st.session_state.live_transcription += " " + transcription
                        st.session_state.transcription_history.append(transcription)

            except Exception as e:
                st.error(f"Error processing audio: {e}")
            finally:
                os.unlink(tmp_file.name)

# Method 3: Chunked Real-Time Processing
st.markdown("---")
st.subheader("⚡ Method 3: Chunked Real-Time Processing")
st.markdown("Record in small chunks for near real-time transcription with your custom model:")

if MIC_RECORDER_AVAILABLE:
    col1, col2, col3 = st.columns(3)

    with col1:
        chunk_duration = st.slider("Chunk Duration (seconds)", 1, 5, 3, key="chunk_duration")

    with col2:
        auto_continue = st.checkbox("Auto-continue recording", value=False, key="auto_continue")

    with col3:
        if st.button("🗑️ Clear Chunks"):
            if "chunk_transcriptions" in st.session_state:
                st.session_state.chunk_transcriptions = []
                st.rerun()

    # Initialize chunk transcriptions
    if "chunk_transcriptions" not in st.session_state:
        st.session_state.chunk_transcriptions = []
    if "chunk_counter" not in st.session_state:
        st.session_state.chunk_counter = 0

    st.info(f"🎙️ Recording {chunk_duration}s chunks. Your Whisper model will process each chunk.")

    # Record chunks
    chunk_audio = mic_recorder(
        start_prompt=f"🎙️ Record {chunk_duration}s Chunk",
        stop_prompt="⏹️ Stop Chunk",
        just_once=not auto_continue,
        use_container_width=True,
        key=f"chunk_recorder_{st.session_state.chunk_counter}",
    )

    if chunk_audio is not None:
        st.session_state.chunk_counter += 1

        # Extract audio data
        audio_bytes = chunk_audio["bytes"]
        sample_rate = chunk_audio.get("sample_rate", 44100)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

        # Show audio player for this chunk
        st.audio(audio_bytes, format="audio/wav")

        # Transcribe with your model
        with st.spinner(
            f"🤖 Processing chunk {len(st.session_state.chunk_transcriptions) + 1} with your Whisper model..."
        ):
            chunk_transcription = transcribe_audio(audio_np, processor, model, device, sample_rate)

        if chunk_transcription:
            # Add to chunk transcriptions
            st.session_state.chunk_transcriptions.append(
                {
                    "chunk": len(st.session_state.chunk_transcriptions) + 1,
                    "text": chunk_transcription,
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            )

            st.success(f"**Chunk {len(st.session_state.chunk_transcriptions)} Transcribed:**")
            st.write(chunk_transcription)

            # Auto-continue if enabled
            if auto_continue:
                time.sleep(0.1)  # Small delay
                st.rerun()

    # Display all chunk transcriptions
    if st.session_state.chunk_transcriptions:
        st.markdown("### 📝 Live Transcription Stream")

        # Create combined transcription
        combined_text = " ".join([chunk["text"] for chunk in st.session_state.chunk_transcriptions])
        st.text_area("Combined Transcription", combined_text, height=150, key="combined_chunks")

        # Show individual chunks
        with st.expander("📋 Individual Chunks", expanded=False):
            for chunk_data in st.session_state.chunk_transcriptions:
                st.markdown(f"**Chunk {chunk_data['chunk']} ({chunk_data['timestamp']}):** {chunk_data['text']}")

        # Download options
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "💾 Download Combined", combined_text, file_name="live_transcription_combined.txt", mime="text/plain"
            )
        with col2:
            detailed_text = "\n\n".join(
                [f"Chunk {c['chunk']} ({c['timestamp']}):\n{c['text']}" for c in st.session_state.chunk_transcriptions]
            )
            st.download_button(
                "💾 Download Detailed", detailed_text, file_name="live_transcription_detailed.txt", mime="text/plain"
            )

else:
    st.warning("Install streamlit-mic-recorder for chunked real-time processing: `pip install streamlit-mic-recorder`")

# Installation instructions
# Method 4: Simulated Real-Time with Threading
st.markdown("---")
st.subheader("🔄 Method 4: Simulated Continuous Mode")
st.markdown("Record continuously and process with your model in batches:")

if "continuous_mode" not in st.session_state:
    st.session_state.continuous_mode = False
if "batch_transcriptions" not in st.session_state:
    st.session_state.batch_transcriptions = []

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶️ Start Continuous Mode", type="primary"):
        st.session_state.continuous_mode = True
        st.rerun()

with col2:
    if st.button("⏹️ Stop Continuous Mode"):
        st.session_state.continuous_mode = False

with col3:
    if st.button("🗑️ Clear Batches"):
        st.session_state.batch_transcriptions = []
        st.rerun()

if st.session_state.continuous_mode:
    st.success("🔴 **CONTINUOUS MODE ACTIVE** - Your Whisper model is ready!")
    st.info("👆 Use the microphone recorder above repeatedly to simulate continuous transcription")

    # Show instructions for continuous mode
    st.markdown("""
    **How to use Continuous Mode:**
    1. Click the microphone button above (Method 1 or 2)
    2. Record your speech (2-5 seconds recommended)
    3. Your custom Whisper model will process it
    4. Repeat for continuous transcription
    5. All transcriptions are combined below
    """)

    # Auto-refresh indicator
    st.markdown("🔄 *Refreshing every 2 seconds to check for new recordings...*")
    time.sleep(2)
    st.rerun()
else:
    st.info("⏸️ Continuous mode stopped")

# Display batch transcriptions
if st.session_state.batch_transcriptions:
    st.markdown("### 📝 Continuous Transcription")
    full_text = " ".join([batch["text"] for batch in st.session_state.batch_transcriptions])
    st.text_area("Full Continuous Transcription", full_text, height=200, key="continuous_display")

    st.download_button(
        "💾 Download Continuous Transcription", full_text, file_name="continuous_transcription.txt", mime="text/plain"
    )

st.markdown("---")
st.subheader("📦 Installation Requirements")

if not MIC_RECORDER_AVAILABLE and not AUDIO_RECORDER_AVAILABLE:
    st.error("No audio recording components available!")


# Show transcription history
if st.session_state.transcription_history:
    st.markdown("---")
    st.subheader("📋 Transcription History")
    for i, trans in enumerate(st.session_state.transcription_history[-10:], 1):
        st.markdown(f"**{i}.** {trans}")

# Footer
st.markdown("---")
st.caption("🤖 Powered by Whisper (Arabic Quran Fine-tuned) & Streamlit")
st.caption("🔐 Authenticated via Hugging Face token")
