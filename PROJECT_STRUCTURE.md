# 📁 Project Structure

```
video_editor_python/
│
├── main.py                    # Main video editor script
├── requirements.txt           # Python dependencies
├── setup_folders.py          # Quick setup script
│
├── README.md                 # Complete documentation
├── QUICKSTART.md             # Quick start guide
├── PROJECT_STRUCTURE.md      # This file
├── .gitignore                # Git ignore rules
│
├── videos/                   # Input videos folder (create this)
│   ├── README.md
│   ├── clip1.mp4
│   ├── clip2.mp4
│   └── ...
│
└── music/                    # Music tracks folder (create this)
    ├── README.md
    ├── epic.mp3
    ├── calm.mp3
    ├── tense.mp3
    ├── joyful.mp3
    └── neutral.mp3
```

## 📝 File Descriptions

### Core Files

- **`main.py`** - The main application
  - AI emotion detection using VideoMAE
  - 6 cinematic filters (dramatic, cool, warm, vintage, dreamy, neutral)
  - Multiple transitions (zoom in/out, crossfades)
  - Smart music selection
  - 720p HD output

- **`requirements.txt`** - Python package dependencies
  - moviepy - video editing
  - transformers - AI models
  - torch/torchvision - deep learning
  - opencv-python - filters
  - scipy, numpy, Pillow - image processing

- **`setup_folders.py`** - Convenience script
  - Creates `videos/` and `music/` folders
  - Adds helpful README files

### Documentation

- **`README.md`** - Complete documentation
  - Features overview
  - Installation instructions
  - Usage examples
  - Troubleshooting
  - Customization guide

- **`QUICKSTART.md`** - 5-minute quick start
  - Step-by-step setup
  - Minimal explanation
  - Common issues

- **`PROJECT_STRUCTURE.md`** - This file
  - Project layout
  - File descriptions

### Input Folders (You Create These)

- **`videos/`** - Place your input videos here
  - Supports: MP4, MOV, AVI, MKV, M4V
  - Need at least 2 videos
  - Recommended: 3-8 videos, 5-10 seconds each

- **`music/`** - Place your music tracks here
  - Must have these exact filenames:
    - `epic.mp3` - for action/sports scenes
    - `calm.mp3` - for peaceful scenes
    - `tense.mp3` - for suspenseful scenes
    - `joyful.mp3` - for happy scenes
    - `neutral.mp3` - for general content
  - Supports: MP3, WAV, M4A

## 🔧 How It Works

### Emotion Detection Pipeline

```
Input Videos → VideoMAE Model → Action Labels → Emotion Mapping → Dominant Emotion
```

**Example:**
```
surfing.mp4 → "surfing water" → "epic"
yoga.mp4 → "yoga" → "calm"
party.mp4 → "celebrating" → "joyful"
→ Dominant: "epic" (most common)
```

### Video Processing Pipeline

```
Load Video → Extract Segment → Resize to 720p → Apply Filter → Add Transitions → Output
```

**Per-clip processing:**
1. Extract 4-second segment from middle
2. Resize to 720p (1280x720)
3. Apply emotion-based filter
4. Add transitions:
   - First clip: fade in + zoom in
   - Middle clips: crossfade
   - Last clip: fade out + zoom out

### Final Assembly

```
All Processed Clips → Concatenate → Add Music → Audio Fade → Render MP4
```

## 🎨 Available Filters

| Filter | Emotion | Description |
|--------|---------|-------------|
| Dramatic | Epic | High contrast, saturated colors |
| Cool Cinematic | Tense | Blue tint, moody atmosphere |
| Warm Cinematic | Joyful | Golden hour, orange/yellow tones |
| Soft Dreamy | Calm | Slight blur, bright and airy |
| Vintage | - | Warm sepia tones, vignette |
| Neutral Enhance | Neutral | Subtle contrast and sharpness |

## 🎬 Transition Types

| Transition | When | Effect |
|------------|------|--------|
| Fade In + Zoom In | First clip | Smooth entrance, zoom from 110% to 100% |
| Crossfade | Middle clips | Overlapping fade between clips |
| Fade Out + Zoom Out | Last clip | Smooth exit, zoom from 100% to 110% |

## 📊 Output Specifications

- **Resolution**: 1280x720 (720p HD)
- **Frame Rate**: 24 fps
- **Video Codec**: H.264 (libx264)
- **Audio Codec**: AAC
- **Bitrate**: 5000k
- **Format**: MP4

## 🔄 Workflow Example

```bash
# 1. Setup
python setup_folders.py

# 2. Add content
cp ~/Downloads/vacation*.mp4 videos/
cp ~/Music/background_music/*.mp3 music/

# 3. Run editor
python main.py --input ./videos

# 4. Output
# → cinematic_output.mp4
```

**Note**: Music folder defaults to `./music` automatically!

## 💡 Customization Points

Want to modify the editor? Here are key areas:

### Change Clip Duration
```python
# Line 283 in main.py
clip_duration = 4.0  # Change to 3.0, 5.0, etc.
```

### Change Output Resolution
```python
# Line 229 in main.py
clip = clip.resize(height=720)  # Change to 1080 for Full HD
```

### Adjust Filter Strength
```python
# Lines 64-141 in main.py
# Each filter function has tunable parameters:
alpha = 1.15  # Contrast (1.0 = normal)
beta = -5     # Brightness (0 = normal)
```

### Add New Emotions
```python
# Line 20 in main.py - KINETICS_TO_EMOTION_MAP
"swimming": "calm",
"boxing": "epic",
# etc.
```

### Modify Transition Duration
```python
# Line 247 in main.py
transition_duration = 1.0  # Change to 0.5, 1.5, etc.
```

## 🎯 Performance Notes

**Processing Time** (approximate, CPU):
- Short clip (4 sec): ~1-2 minutes
- 5 clips total: ~5-10 minutes
- GPU: 3-5x faster

**Memory Usage**:
- Base: ~2GB RAM
- +500MB per 1080p video being processed
- Model: ~400MB (one-time download)

**Disk Space**:
- Model cache: ~400MB
- Temp files during processing: 2-3x input size
- Output: ~1MB per second of video

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| Import errors | `pip install -r requirements.txt` |
| FFmpeg not found | Install FFmpeg for your OS |
| Out of memory | Use fewer/shorter videos |
| Slow processing | Normal on CPU, consider GPU |
| Model download fails | Check internet, wait for download |

---

**Happy editing! 🎬✨**

