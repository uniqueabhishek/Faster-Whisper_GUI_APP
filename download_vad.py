import os
import logging
import urllib.request
import ssl

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("download_vad")

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
VAD_PATH = os.path.join(ASSETS_DIR, "silero_vad.onnx")

def download_vad():
    if not os.path.exists(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)

    if os.path.exists(VAD_PATH):
        try:
            os.remove(VAD_PATH)
            LOGGER.info("Removed existing invalid VAD model.")
        except Exception as e:
            LOGGER.warning(f"Could not remove existing file: {e}")

    # Silero VAD v4 (compatible with installed faster_whisper)
    url = "https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx"

    LOGGER.info(f"Downloading VAD model from {url}...")
    try:
        # Create a context that doesn't verify SSL certs if that's an issue (optional, but helps with some proxies)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': '*/*'
            }
        )

        with urllib.request.urlopen(req, context=ctx) as response, open(VAD_PATH, 'wb') as out_file:
            data = response.read()
            if len(data) < 500000: # < 500KB
                 # Check content
                 try:
                     content = data.decode('utf-8', errors='ignore')
                     if "<html" in content.lower() or "<!doctype" in content.lower():
                         raise RuntimeError("Downloaded file is HTML (likely a block page).")
                 except Exception:  # pylint: disable=broad-except
                     pass
                 raise RuntimeError(f"Downloaded file is too small ({len(data)} bytes).")

            out_file.write(data)

        LOGGER.info(f"Successfully downloaded VAD model to: {VAD_PATH}")
        LOGGER.info(f"File size: {os.path.getsize(VAD_PATH)} bytes")

    except Exception as e:
        LOGGER.error(f"Failed to download VAD model: {e}")
        if os.path.exists(VAD_PATH):
            try:
                os.remove(VAD_PATH)
            except Exception:  # pylint: disable=broad-except
                pass
        raise

if __name__ == "__main__":
    download_vad()
