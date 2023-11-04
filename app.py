import streamlit as st
import whisper
import os
from tempfile import NamedTemporaryFile

# Load the Whisper model
model = whisper.load_model("tiny")

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
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"], accept_multiple_files=False)

# Transcribe button
if st.button("Transcribe"):
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_filename = tmp.name
        
        # Transcribe the file
        with st.spinner('Transcribing...'):
            try:
                result = model.transcribe(tmp_filename, fp16=False)
                transcription = result["text"]
                st.text_area("Transcribed Text:", transcription, height=250)
            except Exception as e:
                st.error(f'An error occurred during transcription: {e}')
            finally:
                # Clean up the temporary file
                os.remove(tmp_filename)
    else:
        st.error("Please upload an audio file first.")
