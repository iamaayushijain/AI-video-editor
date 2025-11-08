# Cinematic Video Editor

An AI-powered video editor that automatically creates cinematic videos with emotion detection, smart transitions, and dramatic effects.

## Features

- 🎬 **AI Emotion Detection** - Automatically detects emotions in videos using VideoMAE
- 🎨 **Cinematic Filters** - Applies emotion-matched filters (dramatic, warm, cool, vintage, etc.)
- ✨ **Smart Transitions** - Explosive openings, dramatic transitions, and emotional endings
- 💫 **Visual Effects** - Confetti, hearts, sparkles, light leaks, and more
- 🎵 **Auto Music Selection** - Matches background music to video emotion
- 📊 **Quality Analysis** - Selects best frames and segments automatically
- 🎥 **1080p Output** - High-quality Full HD videos

## Installation

### 1. Clone or download this repository

```bash
cd video_editor_python
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Note:** This will download large ML models (~500MB) on first run.

## Setup

### 1. Prepare your videos

Put all your input videos in a folder (e.g., `./videos/`):
```
videos/
  ├── video1.mp4
  ├── video2.mp4
  └── video3.mp4
```

### 2. Prepare your music

Put music files in a folder (default: `./music/`):
```
music/
  ├── epic.mp3
  ├── calm.mp3
  ├── joyful.mp3
  ├── tense.mp3
  └── neutral.mp3
```

**Note:** The script will automatically match music to detected emotions.

## Usage

### Basic usage

```bash
python main.py --input ./videos
```

This will:
- Process all videos in `./videos/`
- Use music from `./music/` (default)
- Create `cinematic_output.mp4`

### Custom output file

```bash
python main.py --input ./videos --output my_video.mp4
```

### Custom music folder

```bash
python main.py --input ./videos --music /path/to/music --output my_video.mp4
```

## How It Works

1. **Loads videos** from input folder
2. **Detects emotions** using AI (epic, calm, joyful, tense, neutral)
3. **Finds best segments** using scene detection and quality analysis
4. **Applies filters** matched to emotions
5. **Adds transitions** - explosive openings, dramatic cuts, emotional endings
6. **Adds effects** - sparkles, hearts, confetti, light leaks
7. **Selects music** based on dominant emotion
8. **Renders** a polished 1080p cinematic video

## Output

- **Resolution:** 1080p Full HD
- **Duration:** ~2.5 seconds per video segment
- **Format:** MP4 (H.264)
- **Effects:** Dramatic transitions, visual effects, emotion-matched filters

## Requirements

- Python 3.8+
- FFmpeg (usually installed with moviepy)
- ~2GB disk space for models and processing
- GPU optional (works on CPU, but slower)

