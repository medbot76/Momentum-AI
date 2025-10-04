import os
from elevenlabs.client import ElevenLabs

def generate_teen_voice(text, output_path=None):
    """
    Generate speech from text using ElevenLabs with a teenager-sounding voice.
    Args:
        text (str): The text to convert to speech.
        output_path (str, optional): If provided, saves the audio to this path.
    Returns:
        bytes: The audio data in MP3 format.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY environment variable not set.")
    voice_id = "bMxLr8fP6hzNRRi9nJxU"
    client = ElevenLabs(api_key=api_key)
    # Get the first chunk of audio (the API yields bytes)
    audio_iter = client.text_to_speech.convert(voice_id=voice_id, text=text, output_format="mp3_44100_128")
    audio_bytes = b"".join(audio_iter)
    if output_path:
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
    return audio_bytes 