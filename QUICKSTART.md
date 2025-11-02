# 🚀 Quick Start Guide

Get started with the AI Video Editor in 5 minutes!

## Step 1: Install Dependencies

```bash
# Install FFmpeg (required)
brew install ffmpeg  # macOS
# OR
sudo apt install ffmpeg  # Linux

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt
```

## Step 2: Setup Folders

```bash
# Run the setup script
python setup_folders.py
```

This creates:
- `videos/` - for your input videos
- `music/` - for your music tracks

## Step 3: Add Your Content

### Add Videos
Copy 2 or more video files into the `videos/` folder:
```
videos/
├── clip1.mp4
├── clip2.mp4
└── clip3.mp4
```

### Add Music
Add these 5 music files to the `music/` folder:
```
music/
├── epic.mp3      # Action/sports music
├── calm.mp3      # Peaceful/relaxing music
├── tense.mp3     # Suspenseful music
├── joyful.mp3    # Happy/upbeat music
└── neutral.mp3   # General background music
```

**Note**: The file names must match exactly! The editor selects music based on detected emotions.

## Step 4: Run the Editor

```bash
python main.py --input ./videos
```

**Note**: Music folder defaults to `./music` - no need to specify it!

**First run will download the AI model (~400MB) - this only happens once!**

## Step 5: Wait for Magic ✨

The editor will:
1. ✅ Load your videos
2. 🤖 Analyze emotions using AI
3. 🎵 Select appropriate music
4. 🎨 Apply cinematic filters
5. ✂️ Add smooth transitions
6. 🎬 Render final video

## Output

Find your finished video as `cinematic_output.mp4` in the same folder!

---

## Example with Custom Output Name

```bash
python main.py --input ./videos --output my_awesome_video.mp4
```

## Example with Custom Music Folder

```bash
python main.py --input ./videos --music /path/to/custom/music
```

## Troubleshooting

**"ffmpeg not found"**
→ Install FFmpeg (see Step 1)

**"No module named 'moviepy'"**
→ Run `pip install -r requirements.txt`

**"Model download failed"**
→ Check internet connection, model downloads on first run

**Processing is slow**
→ Normal! Each clip takes 1-2 min on CPU. Use fewer/shorter videos.

---

## Tips

- Use 3-8 video clips for best results
- Input videos should be at least 5-10 seconds long
- Higher quality input = better output
- Mix different scene types for variety

---

**Need more help?** Check the full [README.md](README.md)

