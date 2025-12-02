import os
import urllib.request
import logging

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("download_vc")

VC_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
VC_PATH = os.path.join(ASSETS_DIR, "vc_redist.x64.exe")

def download_vc():
    if not os.path.exists(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)
        LOGGER.info(f"Created assets directory: {ASSETS_DIR}")

    if os.path.exists(VC_PATH):
        LOGGER.info(f"VC++ installer already exists at: {VC_PATH}")
        return

    LOGGER.info(f"Downloading VC++ Redistributable from {VC_URL}...")
    try:
        req = urllib.request.Request(
            VC_URL,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as response, open(VC_PATH, 'wb') as out_file:
            out_file.write(response.read())
        LOGGER.info(f"Successfully downloaded VC++ installer to: {VC_PATH}")
    except Exception as e:
        LOGGER.error(f"Failed to download VC++ installer: {e}")
        raise

if __name__ == "__main__":
    download_vc()
