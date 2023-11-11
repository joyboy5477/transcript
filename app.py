import streamlit as st
import whisper
import os
import moviepy.editor as mp
from pydub import AudioSegment
from tempfile import NamedTemporaryFile

# Load the Whisper model
model = whisper.load_model("base")

st.title("Audio Transcription with OpenAI's Whisper")

st.markdown("""
    <style>
        .block-container{
            padding-top: 2rem;
        }
        .stButton > button {
            width: 100%;
            padding: 0.5rem 0rem;
            margin-top: 1rem;
        }
        .stTextArea > textarea {
            min-height: 250px;
        }
    </style>
""", unsafe_allow_html=True)

# File uploader widget
uploaded_file = st.file_uploader("Upload an audio or video file", type=["wav", "mp3", "ogg", "mp4","m4a"], accept_multiple_files=False)

# Transcribe button
if st.button("Transcribe"):
    if uploaded_file is not None:
        with NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_filename = tmp.name
        
        # Check if the file needs conversion (M4A or MP4)
        if tmp_filename.endswith('.m4a') or tmp_filename.endswith('.mp4'):
            with st.spinner('Processing the audio file...'):
                # Convert M4A to WAV (or MP3)
                if tmp_filename.endswith('.m4a'):
                    audio = AudioSegment.from_file(tmp_filename, "m4a")
                    audio_filename = tmp_filename + '.wav'  # Convert to WAV
                    audio.export(audio_filename, format="wav")

                # Extract audio from MP4
                elif tmp_filename.endswith('.mp4'):
                    video = mp.VideoFileClip(tmp_filename)
                    audio_filename = tmp_filename + '.mp3'  # Convert to MP3
                    video.audio.write_audiofile(audio_filename)

                os.remove(tmp_filename)
                tmp_filename = audio_filename

        # Transcribe the audio
        with st.spinner('Transcribing...'):
            try:
                result = model.transcribe(tmp_filename, fp16=False)
                transcription = result["text"]
                st.text_area("Transcribed Text:", transcription, height=250)
            except Exception as e:
                st.error(f'An error occurred during transcription: {e}')
            finally:
                # Clean up the temporary file
                if os.path.exists(tmp_filename):
                    os.remove(tmp_filename)
    else:
        st.error("Please upload an audio or video file first.")
