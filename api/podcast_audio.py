# api/podcast_audio.py
# Synthesises one dialogue segment to WAV using Gemini's multi-speaker TTS.
# Body: { "dialogue": "Alex: ...\nSam: ..." }  ->  audio/wav (+ X-Audio-Duration)
import io
import json
import os
import sys
import wave
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_flashcards import json_response, resolve_cors_headers

TTS_MODEL = os.environ.get("PODCAST_TTS_MODEL", "gemini-2.5-flash-preview-tts")

# Host name -> Gemini prebuilt voice. Names must match the script generator's.
HOST_VOICES = [
    ("Alex", "Kore"),
    ("Sam", "Puck"),
]


def pcm_to_wav(pcm: bytes, rate: int, channels: int = 1, sampwidth: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def synthesize(dialogue: str):
    """Returns (wav_bytes, duration_seconds).

    Audio generation cannot go through Zen: it proxies text models only, and
    multi-speaker TTS needs the Gemini API directly. A missing key is therefore
    reported as a clear 503 rather than a KeyError traceback.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Podcast audio needs the Gemini API "
            "directly - the Zen proxy serves text models only."
        )

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    speech = types.SpeechConfig(
        multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
            speaker_voice_configs=[
                types.SpeakerVoiceConfig(
                    speaker=name,
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                    ),
                )
                for name, voice in HOST_VOICES
            ]
        )
    )
    config = types.GenerateContentConfig(response_modalities=["AUDIO"], speech_config=speech)
    response = client.models.generate_content(model=TTS_MODEL, contents=dialogue, config=config)

    part = response.candidates[0].content.parts[0]
    inline = getattr(part, "inline_data", None)
    if inline is None:
        raise RuntimeError("No audio in response")
    pcm = inline.data
    mime = getattr(inline, "mime_type", "") or ""

    rate = 24000
    if "rate=" in mime:
        try:
            rate = int(mime.split("rate=")[1].split(";")[0])
        except ValueError:
            pass

    wav = pcm_to_wav(pcm, rate)
    duration = len(pcm) / (rate * 2)  # 16-bit mono
    return wav, duration


def lambda_handler(event):
    try:
        if event["httpMethod"] != "POST":
            return json_response(405, {"success": False, "error": "Method not allowed"})

        try:
            body = json.loads(event["body"])
            dialogue = str(body.get("dialogue", "")).strip()
        except (TypeError, ValueError):
            return json_response(400, {"success": False, "error": "Malformed request body"})

        if not dialogue:
            return json_response(400, {"success": False, "error": "Missing dialogue"})
        if len(dialogue) > 6000:
            return json_response(400, {"success": False, "error": "Dialogue too long"})

        try:
            wav, duration = synthesize(dialogue)
        except Exception as exc:
            print("TTS failed:", repr(exc))
            message = str(exc)
            if "GEMINI_API_KEY" in message:
                return json_response(503, {"success": False,
                                           "error": "Podcast audio is not configured",
                                           "details": message})
            return json_response(502, {"success": False, "error": "TTS failed",
                                       "details": message})

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "audio/wav",
                "Content-Length": str(len(wav)),
                "X-Audio-Duration": "%.2f" % duration,
            },
            "body": wav,
        }

    except Exception as e:
        import traceback
        print("Unhandled error:", str(e))
        print(traceback.format_exc())
        return json_response(500, {"success": False, "error": str(e)})


class handler(BaseHTTPRequestHandler):
    def _send(self, response):
        body = response["body"]
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(response["statusCode"])
        for k, v in response.get("headers", {}).items():
            self.send_header(k, v)
        if "Content-Length" not in response.get("headers", {}):
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        event = {"httpMethod": "POST", "headers": dict(self.headers), "body": body}
        response = lambda_handler(event)
        response.setdefault("headers", {}).update(resolve_cors_headers(event))
        self._send(response)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
