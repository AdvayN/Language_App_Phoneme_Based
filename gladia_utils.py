import requests
import os 
import mimetypes
import time
import streamlit as st


GLADIA_UPLOAD_URL = "https://api.gladia.io/v2/upload"
GLAIDA_TRANSCRIPTION_URL = "https://api.gladia.io/v2/pre-recorded"
TIME_INTERVAL = 2
TOTAL_REQUESTS = 10


# Gladia api key
os.environ["GLADIA_API_KEY"] = st.secrets["GLADIA_API_KEY"]


def upload_file_to_gladia(file_path: str, api_key: str = os.environ["GLADIA_API_KEY"]) -> dict:
    # set the api key
    headers = {"x-gladia-key": api_key}

    with open(file_path, 'rb') as audio_file:
        # set the mime type
        mime = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        # file setup
        files = {"audio": (os.path.basename(file_path), audio_file, mime)}
        # send response
        try:
            response = requests.post(GLADIA_UPLOAD_URL, files=files, headers=headers)
            api_response_url = response.json()["audio_url"]
            return {
                "audio_url": api_response_url
            }
        except Exception as e:
            print(f"Error uploading file to Gladia: {e}")
            return {
                "audio_url": None
            }


def transcribe_audio(audio_url: str, api_key: str = os.environ["GLADIA_API_KEY"], **options) -> dict:
    """
    Sends a transcription request for a previously uploaded audio file.

    :param audio_url: The URL returned from the upload step.
    :param api_key: Your Gladia API key.
    :param options: Optional transcription parameters (e.g., diarization, translation).
    :return: JSON containing transcription ID and result_url.
    """
    headers = {
        "x-gladia-key": api_key,
        "Content-Type": "application/json"
    }
    data = {"audio_url": audio_url}
    data.update(options)  # Add diarization, translation, etc.

    try:
        response = requests.post(GLAIDA_TRANSCRIPTION_URL, headers=headers, json=data)
        data_id = response.json()["id"]
        return {
            "transcription_id": data_id,
            
        }
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return {
            "transcription_id": None,
        }


def get_transcription_result(transcription_id: str, api_key: str = os.environ["GLADIA_API_KEY"]) -> dict:
    """
    Retrieves the result of a transcription request.

    :param transcription_id: The ID returned from the transcribe_audio step.
    :param api_key: Your Gladia API key.
    :return: JSON containing the transcription result.
    """

    url = f"{GLAIDA_TRANSCRIPTION_URL}/{transcription_id}"
    headers = {
        "x-gladia-key": api_key,
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(url, headers=headers)
        trsncript_response = response.json()
        if trsncript_response["status"] == "done":
            return {
                "transcript": trsncript_response["result"]["transcription"]["utterances"],
                "status": "done",
            }
        elif trsncript_response["status"] == "processing":
            return {
                "transcript": None,
                "status": "processing",
            }
        elif trsncript_response["status"] == "queued":
            return {
                "transcript": None,
                "status": "queued",
            }
    except Exception as e:
        print(f"Error getting transcription result: {e}")
        return {
            "transcript": None,
            "status": "failed",
        }


def poll_transcription(transcription_id: str) -> dict:
    request_count = 0
    while True:
        request_count += 1
        result = get_transcription_result(transcription_id)
        if result["status"] == "done":
            return result
        elif result["status"] == "processing":
            time.sleep(TIME_INTERVAL)
        elif result["status"] == "queued":
            time.sleep(TIME_INTERVAL)
        else:
            return result
        # if polling persists, break the loop
        if request_count == TOTAL_REQUESTS:
            return result