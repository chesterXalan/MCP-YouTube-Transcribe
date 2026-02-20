import logging
import os
import subprocess
import json
import multiprocessing
import urllib.request
import whisper
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from yt_dlp.utils import DownloadError
from pytube import YouTube
from pytube.exceptions import PytubeError

# This will get the logger that was configured in mcp_server.py
logger = logging.getLogger(__name__)

# Supported Whisper model names (compatible with both whisper.cpp and Python Whisper)
VALID_MODELS = ("tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo")

# Mapping from model name to whisper.cpp ggml filename
WHISPER_CPP_FILENAMES = {name: f"ggml-{name}.bin" for name in VALID_MODELS}

# HuggingFace base URL for downloading whisper.cpp models
HUGGINGFACE_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

# A simple cache for the Whisper model so we don't reload it every time
_whisper_model = None
_whisper_model_name: str | None = None


def _get_config() -> dict:
    """Read Whisper configuration from environment variables."""
    model_name = os.getenv("WHISPER_MODEL", "tiny").strip()
    if model_name not in VALID_MODELS:
        raise ValueError(
            f"Invalid WHISPER_MODEL '{model_name}'. Valid options: {', '.join(VALID_MODELS)}"
        )

    model_dir = os.getenv("WHISPER_MODEL_DIR", "").strip()
    if not model_dir:
        model_dir = os.path.join(os.path.dirname(__file__), "models")

    language = os.getenv("WHISPER_LANGUAGE", "auto").strip()

    return {"model_name": model_name, "model_dir": model_dir, "language": language}


def _download_model(model_name: str, model_dir: str) -> str:
    """
    Ensure the whisper.cpp model file exists, downloading from HuggingFace if needed.
    Returns the absolute path to the model file.
    """
    filename = WHISPER_CPP_FILENAMES[model_name]
    model_path = os.path.join(model_dir, filename)

    if os.path.exists(model_path):
        return model_path

    os.makedirs(model_dir, exist_ok=True)
    url = f"{HUGGINGFACE_BASE_URL}/{filename}"
    tmp_path = model_path + ".downloading"

    logger.info(f"Model '{model_name}' not found at {model_path}. Downloading from {url} ...")

    def _progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            percent = block_num * block_size * 100 / total_size
            # Log every ~10%
            if int(percent) % 10 == 0 and int(percent) != int((block_num - 1) * block_size * 100 / total_size):
                logger.info(f"Downloading {model_name}: {min(percent, 100):.0f}%")

    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook=_progress_hook)
        os.rename(tmp_path, model_path)
        logger.info(f"Model '{model_name}' downloaded successfully to {model_path}")
        return model_path
    except Exception as e:
        # Clean up partial download
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Failed to download model '{model_name}' from {url}: {e}") from e


def _check_whisper_cpp():
    """Check if whisper.cpp is available in the system PATH."""
    try:
        result = subprocess.run(['whisper-cli', '--help'],
                                capture_output=True,
                                text=True)
        logger.info("whisper.cpp check result: %s", result.returncode == 0)
        return result.returncode == 0
    except FileNotFoundError:
        logger.info("whisper.cpp not found in PATH")
        return False


def _transcribe_with_whisper_cpp(audio_path, model_name: str = "tiny", model_dir: str = "", language: str = "auto"):
    """
    Transcribe audio using whisper.cpp.
    Returns the transcript text if successful, None if failed.
    """
    try:
        logger.info("Attempting transcription with whisper.cpp...")

        # Try to get (or download) the model
        try:
            model_path = _download_model(model_name, model_dir)
        except (RuntimeError, KeyError) as e:
            logger.error(f"Could not obtain whisper.cpp model: {e}")
            return None

        # Convert MP3 to WAV
        wav_path = audio_path.replace('.mp3', '.wav')
        logger.info(f"Converting {audio_path} to WAV format...")
        convert_result = subprocess.run(
            ['ffmpeg', '-y', '-i', audio_path, wav_path],
            capture_output=True,
            text=True
        )
        if convert_result.returncode != 0:
            logger.error(f"Failed to convert audio to WAV: {convert_result.stderr}")
            return None

        # Run whisper.cpp on the WAV file
        json_output_path = wav_path + '.json'

        # Determine thread count
        thread_env = os.getenv("WHISPER_THREADS")
        try:
            threads = int(thread_env) if thread_env else multiprocessing.cpu_count()
        except ValueError:
            threads = multiprocessing.cpu_count()

        cmd = [
            'whisper-cli',
            '-m', model_path,
            '-t', str(threads),  # multi-threading
            '--language', language,
            '--output-json-full',
            '--no-timestamps',
            wav_path
        ]
        logger.info(f"Running whisper.cpp command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        logger.info(f"whisper.cpp return code: {result.returncode}")
        logger.info(f"whisper.cpp stderr: {result.stderr}")

        # Clean up the WAV file
        try:
            os.remove(wav_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary WAV file: {e}")

        if result.returncode == 0:
            try:
                # Read the JSON output from the file
                with open(json_output_path, 'r', encoding='utf-8', errors='replace') as f:
                    output = json.load(f)

                # Clean up the JSON file
                try:
                    os.remove(json_output_path)
                except Exception as e:
                    logger.warning(f"Failed to remove JSON output file: {e}")

                # Extract text from the full JSON output
                segments = output.get('transcription', [])
                text = ' '.join(seg.get('text', '') for seg in segments)
                return text.strip()
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.error(f"Failed to read whisper.cpp JSON output: {e}")
                return None
        else:
            logger.error(f"whisper.cpp failed with error: {result.stderr}")
            return None
    except Exception as e:
        logger.error(f"Error running whisper.cpp: {e}")
        return None


def _yt_dlp_hook(d):
    """
    A hook for yt-dlp to capture its progress. We use this to prevent it from
    printing directly to stdout, which would corrupt the JSON-RPC communication.
    """
    if d['status'] == 'finished':
        # The 'filename' key is provided once the download is complete.
        logger.info(f"yt-dlp hook: Finished downloading '{d.get('filename', 'unknown file')}'")
    # We ignore the 'downloading' status to prevent log spam.


def _download_audio_with_fallbacks(video_url: str, audio_path: str) -> bool:
    """
    Tries to download audio, first with yt-dlp, then with pytube as a fallback.
    Returns True if download is successful, False otherwise.
    """
    # --- Attempt 1: yt-dlp (Primary Method) ---
    try:
        logger.info(f"Attempting audio download with yt-dlp...")
        ydl_opts_audio = {
            'format': 'bestaudio/best',
            'outtmpl': audio_path,
            'quiet': True,
            'noprogress': True,
            'overwrites': True,
            'logger': logger,
            'progress_hooks': [_yt_dlp_hook],
        }
        with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
            ydl.download([video_url])
        logger.info("Audio download with yt-dlp complete.")
        return True
    except DownloadError as e:
        logger.warning(f"yt-dlp failed to download audio: {e}. Falling back to pytube.")
    except Exception as e:
        logger.error(f"An unexpected error occurred with yt-dlp: {e}. Falling back to pytube.")

    # --- Attempt 2: pytube (Fallback Method) ---
    try:
        logger.info("Attempting audio download with pytube...")
        yt = YouTube(video_url)
        # Get the best available audio stream (usually .mp4 container with audio)
        audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
        if not audio_stream:
            logger.error("Pytube could not find any audio-only streams.")
            return False

        # pytube downloads to a file with its own name, so we download and then rename if needed.
        # Or, we can specify the filename directly.
        output_dir = os.path.dirname(audio_path)
        file_name = os.path.basename(audio_path)

        audio_stream.download(output_path=output_dir, filename=file_name)
        logger.info("Audio download with pytube complete.")
        return True
    except PytubeError as e:
        logger.error(f"Pytube also failed to download audio: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred with pytube: {e}")
        return False


def get_youtube_transcript(query: str, force_whisper: bool = False, language: str | None = None) -> dict:
    """
    Searches for a YouTube video, downloads it, and returns the transcript.
    It will try to get an official transcript first, unless force_whisper is True.
    If force_whisper is True, it will try whisper.cpp first, then fall back to Python whisper.
    """
    global _whisper_model, _whisper_model_name

    config = _get_config()
    effective_language = language or config["language"]

    logger.info(
        f"get_youtube_transcript called with query: '{query}', force_whisper: {force_whisper}, "
        f"language: {effective_language}, model: {config['model_name']}"
    )

    try:
        # --- 1. Search for the video and get its info (using yt-dlp for robust search) ---
        logger.info("Searching for video with yt-dlp to get info...")
        ydl_opts_info = {
            'format': 'bestaudio/best',
            'noplaylist': True,
            'default_search': 'ytsearch',
            'quiet': True,
            'noprogress': True,
            'logger': logger,
        }
        with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
            info = ydl.extract_info(query, download=False)
            video_info = info['entries'][0] if 'entries' in info else info

        video_url = video_info.get("webpage_url")
        video_title = video_info.get("title", "Unknown Title")
        video_id = video_info.get("id")  # Get video ID for transcript API
        logger.info(f"Found video: '{video_title}' (ID: {video_id}) at {video_url}")

        transcript_text = None
        transcript_source = "Not Available"

        # --- 2. Try to get the official transcript FIRST (unless forced to Whisper) ---
        if not force_whisper and video_id:
            logger.info("Attempting to fetch official YouTube transcript...")
            try:
                api = YouTubeTranscriptApi()
                transcript_list = api.list(video_id)
                # Prioritize manually created transcripts, then auto-generated
                transcript_obj = None
                for t in transcript_list:
                    if t.is_generated:
                        # Keep auto-generated as a fallback, but prefer manual
                        if transcript_obj is None:
                            transcript_obj = t
                    else:
                        # Found a manual transcript, use it immediately
                        transcript_obj = t
                        break  # Found best option, exit loop

                if transcript_obj:
                    # Fetch the actual transcript using the selected language
                    fetched = api.fetch(video_id, languages=[transcript_obj.language_code])
                    transcript_text = " ".join([snippet.text for snippet in fetched])
                    transcript_source = f"Official YouTube Captions ({'Generated' if transcript_obj.is_generated else 'Manual'})"
                    logger.info("Successfully fetched official YouTube transcript.")
                else:
                    logger.info("No official transcript found for this video.")

            except NoTranscriptFound:
                logger.info("No official transcript found for this video (NoTranscriptFound exception).")
            except TranscriptsDisabled:
                logger.info("Transcripts are disabled for this video.")
            except Exception as e:
                logger.warning(f"Error fetching official transcript: {e}", exc_info=True)

        if transcript_text:
            return {
                "status": "success",
                "title": video_title,
                "url": video_url,
                "source": transcript_source,
                "transcript": transcript_text
            }

        # --- 3. If no official transcript or force_whisper, use Whisper (audio download + transcribe) ---
        # This block only runs if transcript_text is still None after the official transcript attempt
        logger.info(
            "Proceeding to audio download and Whisper transcription as no official transcript was available or force_whisper is True.")
        output_dir = os.path.join(os.path.dirname(__file__), "testing", "audio_cache")
        os.makedirs(output_dir, exist_ok=True)
        audio_path = os.path.join(output_dir, f"{video_info.get('id', 'default_id')}.mp3")

        logger.info(f"Downloading audio to: {audio_path}")
        download_successful = _download_audio_with_fallbacks(video_url, audio_path)

        if not download_successful:
            message = "Failed to download audio using all available methods (yt-dlp, pytube)."
            logger.error(message)
            return {"status": "error", "message": message}

        # First try whisper.cpp if available
        if _check_whisper_cpp():
            transcript_text = _transcribe_with_whisper_cpp(
                audio_path,
                model_name=config["model_name"],
                model_dir=config["model_dir"],
                language=effective_language,
            )
            if transcript_text:
                transcript_source = "whisper.cpp (AI Generated)"
                logger.info("Successfully transcribed using whisper.cpp")

        # Fall back to Python whisper if whisper.cpp failed or isn't available
        if transcript_text is None:
            logger.info("Falling back to Python whisper...")
            transcript_source = "Python Whisper (AI Generated)"

            target_model = config["model_name"]
            if _whisper_model is None or _whisper_model_name != target_model:
                logger.info(f"Loading Python Whisper model '{target_model}'...")
                _whisper_model = whisper.load_model(target_model)
                _whisper_model_name = target_model
                logger.info(f"Whisper model '{target_model}' loaded successfully.")

            # Python Whisper uses None for auto-detect, not the string "auto"
            whisper_lang = None if effective_language == "auto" else effective_language

            logger.info("Starting Python Whisper transcription...")
            result = _whisper_model.transcribe(audio_path, fp16=False, language=whisper_lang)
            transcript_text = result["text"]
            logger.info("Python Whisper transcription complete.")

        return {
            "status": "success",
            "title": video_title,
            "url": video_url,
            "source": transcript_source,
            "transcript": transcript_text
        }

    except DownloadError as e:
        logger.error(f"yt-dlp download error during info extraction: {e}", exc_info=True)
        return {"status": "error", "message": f"Could not find video info or initial download failed: {e}"}
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_youtube_transcript: {e}", exc_info=True)
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}
