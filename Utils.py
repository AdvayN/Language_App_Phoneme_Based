import streamlit as st


# function to save the audio as a wav
def save_audio_wav(audio_bytes, file_name):
    with open(file_name, 'wb') as new_wav_file:
        new_wav_file.write(audio_bytes)