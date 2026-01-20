import subprocess
from pathlib import Path

LINK_FILE = "link_video.txt"
OUT_DIR = Path("data_ytb/nu_v2")

OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(LINK_FILE) as f:
    links = [l.strip() for l in f if l.strip()]

for i, url in enumerate(links, 1):
    print(f"⬇  [{i}/{len(links)}] Downloading: {url}")

    cmd = [
        "yt-dlp",
        "-f", "bestaudio/best",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", "ffmpeg:-ac 1 -ar 44100",
        "--no-playlist",
        "-o", str(OUT_DIR / "%(id)s.%(ext)s"),
        url
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f" Failed: {url}")

print(" DONE")