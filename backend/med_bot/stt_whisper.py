import whisper
import tempfile
import os
from pydub import AudioSegment
import io
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperSTT:
    def __init__(self, model_size="base"):
        """
        Initialize Whisper STT with specified model size.
        Available sizes: tiny, base, small, medium, large
        """
        try:
            self.model = whisper.load_model(model_size)
            logger.info(f"Whisper model '{model_size}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe_audio(self, audio_data, audio_format="webm"):
        """
        Transcribe audio data to text.
        
        Args:
            audio_data (bytes): Raw audio data
            audio_format (str): Format of the audio data (webm, wav, mp3, etc.)
        
        Returns:
            str: Transcribed text
        """
        temp_file_path = None
        wav_path = None
        
        try:
            logger.info(f"Processing audio: format={audio_format}, size={len(audio_data)} bytes")
            
            # Create a temporary file to store the audio
            with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            logger.info(f"Temporary file created: {temp_file_path}")
            
            # Convert audio to WAV format for better Whisper compatibility
            wav_path = temp_file_path.replace(f".{audio_format}", ".wav")
            
            try:
                # Use pydub for audio conversion
                if audio_format.lower() in ['webm', 'ogg']:
                    # For WebM, we might need to use ffmpeg directly
                    try:
                        audio = AudioSegment.from_file(temp_file_path, format="webm")
                    except Exception as e:
                        logger.warning(f"pydub failed, trying ffmpeg: {e}")
                        # Fallback to ffmpeg
                        subprocess.run([
                            'ffmpeg', '-i', temp_file_path, 
                            '-ar', '16000',  # 16kHz sample rate
                            '-ac', '1',      # Mono
                            '-c:a', 'pcm_s16le',  # PCM format
                            wav_path, '-y'   # Overwrite output
                        ], check=True, capture_output=True)
                        logger.info("Audio converted using ffmpeg")
                else:
                    audio = AudioSegment.from_file(temp_file_path)
                
                if 'audio' in locals():
                    # Convert to WAV with optimal settings for Whisper
                    audio = audio.set_frame_rate(16000).set_channels(1)
                    audio.export(wav_path, format="wav")
                    logger.info("Audio converted using pydub")
                
            except Exception as conversion_error:
                logger.error(f"Audio conversion failed: {conversion_error}")
                # Try to use the original file if conversion fails
                wav_path = temp_file_path
            
            # Clean up original temp file if we converted it
            if wav_path != temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            
            # Transcribe the audio
            logger.info(f"Starting transcription of: {wav_path}")
            result = self.model.transcribe(wav_path, language="en")
            transcribed_text = result["text"].strip()
            
            logger.info(f"Transcription successful: '{transcribed_text[:50]}...'")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            raise Exception(f"Error transcribing audio: {str(e)}")
        finally:
            # Clean up temporary files
            for file_path in [temp_file_path, wav_path]:
                if file_path and os.path.exists(file_path):
                    try:
                        os.unlink(file_path)
                        logger.info(f"Cleaned up: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up {file_path}: {e}")
    
    def transcribe_file(self, file_path):
        """
        Transcribe audio file to text.
        
        Args:
            file_path (str): Path to the audio file
        
        Returns:
            str: Transcribed text
        """
        try:
            logger.info(f"Transcribing file: {file_path}")
            result = self.model.transcribe(file_path)
            return result["text"].strip()
        except Exception as e:
            logger.error(f"Error transcribing file {file_path}: {str(e)}")
            raise Exception(f"Error transcribing file: {str(e)}")

# Global instance
whisper_stt = WhisperSTT()

def transcribe_audio(audio_data, audio_format="webm"):
    """
    Convenience function to transcribe audio data.
    """
    return whisper_stt.transcribe_audio(audio_data, audio_format)

def transcribe_file(file_path):
    """
    Convenience function to transcribe audio file.
    """
    return whisper_stt.transcribe_file(file_path) 