# 🎬 AI-Powered Cinematic Video Editor

An intelligent video editor that automatically analyzes your video clips, detects their emotional themes, and creates a polished cinematic video with appropriate filters, transitions, and background music.

## ✨ Features

- **AI Emotion Detection**: Uses VideoMAE model to analyze video content and detect emotional themes (epic, calm, tense, joyful, neutral)
- **Cinematic Filters**: Multiple professional-grade filters including:
  - Dramatic high-contrast filter for epic scenes
  - Soft dreamy filter for calm moments
  - Cool cinematic filter for tense scenes
  - Warm golden-hour filter for joyful content
  - Enhanced neutral filter for balanced look
- **Dynamic Transitions**: 
  - Zoom-in effect for opening clip
  - Zoom-out effect for closing clip
  - Smooth crossfades between clips
- **Smart Music Selection**: Automatically selects background music based on detected emotional theme
- **Professional Output**: 720p HD video with optimized encoding

## 📋 Requirements

- Python 3.8 or higher
- FFmpeg (required by moviepy)
- At least 4GB RAM
- GPU recommended for faster processing (but not required)

## 🚀 Installation

### 1. Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

### 2. Clone or Download This Project

```bash
cd video_editor_python
```

### 3. Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Note**: First run will download the VideoMAE model (~400MB) - this only happens once.

## 📁 Folder Structure

Create the following folder structure:

```
video_editor_python/
├── main.py
├── requirements.txt
├── README.md
├── videos/              # Your input videos go here
│   ├── clip1.mp4
│   ├── clip2.mp4
│   └── clip3.mp4
└── music/               # Your music tracks go here
    ├── epic.mp3
    ├── calm.mp3
    ├── tense.mp3
    ├── joyful.mp3
    └── neutral.mp3
```

## 🎵 Music Files

You need to provide music files in the `music/` folder. The files should be named according to the emotion they represent:

- **epic.mp3** - High-energy music for action/sports scenes
- **calm.mp3** - Peaceful music for serene/relaxing scenes  
- **tense.mp3** - Suspenseful music for intense scenes
- **joyful.mp3** - Happy upbeat music for cheerful scenes
- **neutral.mp3** - Balanced music for general content

**Supported formats**: MP3, WAV, M4A

### Where to Get Music

- [YouTube Audio Library](https://www.youtube.com/audiolibrary) (Free)
- [Free Music Archive](https://freemusicarchive.org/) (Free)
- [Epidemic Sound](https://www.epidemicsound.com/) (Subscription)
- [Artlist](https://artlist.io/) (Subscription)

## 🎬 Usage

### Basic Usage

```bash
python main.py --input ./videos
```

The music folder defaults to `./music` in the current directory.

### Custom Output Filename

```bash
python main.py --input ./videos --output my_awesome_video.mp4
```

### Custom Music Folder

```bash
python main.py --input ./videos --music /path/to/custom/music
```

### Full Options

```bash
python main.py \
  --input /path/to/videos \
  --music /path/to/custom/music \
  --output final_video.mp4
```

## 📊 How It Works

1. **Video Loading**: Scans the input folder for all video files (MP4, MOV, AVI, MKV)

2. **AI Analysis**: Each video is analyzed using a pre-trained VideoMAE model that:
   - Extracts 16 frames from each clip
   - Classifies the action/scene type
   - Maps it to an emotional theme

3. **Music Selection**: Based on the dominant emotion across all clips, appropriate background music is selected

4. **Video Processing**: Each clip is:
   - Trimmed to 4 seconds (middle section)
   - Resized to 720p
   - Enhanced with appropriate cinematic filter
   - Given smooth transitions (fade, zoom)

5. **Final Assembly**:
   - All clips are concatenated
   - Background music is added and looped to match video length
   - Audio is faded out at the end
   - Final video is rendered in H.264 format

## ⚙️ Customization

### Adjust Clip Duration

Edit line 283 in `main.py`:

```python
clip_duration = 4.0  # Change to desired seconds
```

### Change Output Resolution

Edit line 229 in `main.py`:

```python
clip = clip.resize(height=720)  # Change to 1080 for Full HD
```

### Modify Filters

All filter functions are defined starting at line 64. You can:
- Adjust contrast/brightness values
- Change color tinting
- Add new filters

### Add More Emotions

Edit the `KINETICS_TO_EMOTION_MAP` dictionary (starting line 20) to map more actions to emotions.

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'moviepy'"
- Make sure you installed requirements: `pip install -r requirements.txt`

### "FileNotFoundError: ffmpeg not found"
- Install FFmpeg (see Installation section)

### "Out of Memory" Error
- Reduce number of input videos
- Lower output resolution
- Close other applications

### "Model download failed"
- Check internet connection
- The model downloads automatically on first run (~400MB)
- If behind proxy, set: `export HF_ENDPOINT=https://hf-mirror.com`

### Video Processing is Slow
- Normal for CPU processing
- Each 4-second clip takes about 1-2 minutes on CPU
- GPU dramatically speeds up processing
- Consider using shorter clips or fewer videos

## 🎯 Tips for Best Results

1. **Video Quality**: Use high-quality input videos (720p or 1080p)
2. **Clip Length**: Input videos should be at least 5-10 seconds long
3. **Variety**: Mix different types of scenes for more dynamic output
4. **Music**: Choose music that matches the theme of your videos
5. **Number of Clips**: 3-8 clips work best for a cohesive short video

## 📝 Example Output

```
Input: 5 video clips of surfing, snowboarding, and rock climbing
Analysis: Detected "epic" as dominant emotion
Music: Selected "epic.mp3"
Filters Applied: Dramatic high-contrast filter
Transitions: Zoom-in (start), crossfades (middle), zoom-out (end)
Output: 20-second cinematic video with epic background music
```

## 🤝 Contributing

Feel free to enhance this project! Some ideas:
- Add more filter styles
- Implement more transition types
- Support for text overlays
- Color grading presets
- Audio ducking for voice-overs

## 📄 License

This project is for educational purposes. Make sure you have rights to any videos and music you use.

## 🙏 Credits

- **VideoMAE Model**: MCG-NJU/videomae-base-finetuned-kinetics
- **MoviePy**: Video editing library
- **OpenCV**: Computer vision filters
- **Transformers**: Hugging Face library

---

**Enjoy creating cinematic videos! 🎥✨**

