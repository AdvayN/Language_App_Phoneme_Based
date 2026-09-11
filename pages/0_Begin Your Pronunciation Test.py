import streamlit as st
import fitz
from Utils import save_audio_wav
import gladia_utils as gut
import references as ref
import evaluation_utils as eu
import os
def extract_text_streamlit(uploaded_file):
    """Extracts text using PyMuPDF from a Streamlit UploadedFile object."""
    try:
        text_content = []

        # Read the bytes from the uploaded file
        file_bytes = uploaded_file.read()

        # Open the document from memory (stream)
        # 'stream' contains the bytes, 'filetype' tells fitz it's a pdf
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text_content.append(page.get_text())

        full_text = "\n".join(text_content)

        # Filter out empty/scanned books (less than 50 chars of text)
        if len(full_text.strip()) < 50:
            return None

        return full_text
    except Exception as e:
        return f"ERROR: {e}"
option = st.selectbox("Select An Option", ["Upload a txt file","Choose a file"])
if option == "Upload a txt file":
    reftxtfile = st.file_uploader("Choose a reference text file", type=['txt','pdf'])
    if reftxtfile is not None:
     # Read file as text
        if reftxtfile.type == 'application/pdf':
            text = extract_text_streamlit(reftxtfile)
        else:
            text = reftxtfile.read().decode("utf-8")
        reference = f'"""\n{text}\n"""'
    else:
        st.warning("Please Upload a txt file")
        st.stop()
elif option== "Choose a file":
    # column for selecting test level and test type
    col_one, col_two, col_three = st.columns(3)

    with col_one:
        test_level = st.selectbox("Select Test Level", ["Level 0","Level 1","Level 2","Level 3"])

    with col_two:
        root_dir = os.path.join("text files",test_level)
        folders = [ item for item in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, item)) ]
        test_type = st.selectbox("Select Test Type", folders)
    
    with col_three:
        root_dir1 = os.path.join(root_dir,test_type)
        files = [ item for item in os.listdir(root_dir1) if os.path.isfile(os.path.join(root_dir1, item)) ]
        Test = st.selectbox("Select Test", files)
        file_path = os.path.join(root_dir1,Test)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        reference = f'"""\n{text}\n"""'

    # # decide the reference type
    # if test_level=="Level 5":
    #     reference = ref.Level_5[test_type]
    # else:
    #     None
    # if not reference:
    #     st.error("Invalid reference type selected!", icon = "🚨")
    #     st.stop()
    
# REFERENCE_MAPPER = {
#     "SWO": ref.SWO,
#     "BLO": ref.BLO,
#     "BOOK": None
# }

# REFERENCE_MAPPER = {
#     "SWO": wrapped_text,
#     "BLO": ref.Level_5["BLO"],
#     "BOOK": None
# }

# add the title and emojis
st.title("🎯 Pronunciation Test")
audio_option = st.selectbox("Select An Option", ["Upload an audio file","Record Audio"])
if audio_option == "Upload an audio file":
    # upload an audio file
    uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3', 'ogg'])
elif audio_option == "Record Audio":
    uploaded_file = st.audio_input("Record your audio")

if uploaded_file is not None:
    st.write(f"Uploaded file: {uploaded_file.name}")

    # Define output file path
    output_path = "converted_file.wav"

    # Save uploaded file as .wav
    save_audio_wav(uploaded_file.read(), output_path)
    # start processing the audio for pronounciation
    with st.spinner("Testing Pronounciation....."):
        # add a toast for processing
        st.toast("Processing Audio for Pronounciation", icon = "🚀")
        # upload the file to gladia
        audio_upload_response = gut.upload_file_to_gladia(output_path)
        audio_url = audio_upload_response["audio_url"]
        if not audio_url:
            # error with an emoji
            st.error("Audio upload failed. Please try again!", icon = "🚨")
            st.stop()
        # initiate the transcribe job
        audio_id_response = gut.transcribe_audio(audio_url)
        audio_id = audio_id_response["transcription_id"]
        if not audio_id:
            # error with an emoji
            st.error("Audio transcription failed. Please try again!", icon = "🚨")
            st.stop()
        # get the transcription
        transcript_response = gut.poll_transcription(audio_id)
        audio_transcript = transcript_response.get("transcript", None)
        #st.success(audio_transcript)
        audio_status = transcript_response.get("status", None)
        if not audio_transcript:
            # error with an emoji
            st.error(f"Audio transcription failed. Please try again! Audio status: {audio_status}", icon = "🚨")
            st.stop()
        # do the evaluation
        evaluation = eu.evaluate_pronounciations(audio_transcript, reference)
        # add a toast for evaluation completed
        st.toast("Evaluating Pronounciation Completed", icon = "🚀")
        # display the evaluation more attractively
        # header with emoji
        st.subheader("🎯 Evaluation Results")
        #st.success(evaluation)
        st.dataframe(evaluation, width="stretch")
    st.success("Evaluation Completed")
    # download button to download the evaluation csv
    st.download_button(
        label="Download Evaluation CSV",
        data=evaluation.to_csv(index=False),
        file_name="evaluation.csv",
        mime="text/csv",
        on_click="ignore"
    )





