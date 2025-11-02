#!/usr/bin/env python3
"""
Quick setup script to create necessary folders for the video editor
"""

import os

def setup_folders():
    """Create necessary folder structure"""
    folders = ['videos', 'music']
    
    print("Setting up folder structure...")
    
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"✅ Created folder: {folder}/")
        else:
            print(f"ℹ️  Folder already exists: {folder}/")
    
    # Create placeholder README in each folder
    video_readme = """# Videos Folder

Place your input video files here.

Supported formats: MP4, MOV, AVI, MKV, M4V

Example:
- clip1.mp4
- clip2.mp4
- clip3.mp4

The editor will process all video files in this folder.
"""
    
    music_readme = """# Music Folder

Place your background music files here.

Required files (name them exactly as shown):
- epic.mp3    (High-energy music for action/sports scenes)
- calm.mp3    (Peaceful music for serene scenes)
- tense.mp3   (Suspenseful music for intense scenes)
- joyful.mp3  (Happy upbeat music)
- neutral.mp3 (Balanced music for general content)

Supported formats: MP3, WAV, M4A

The editor will automatically select music based on detected emotions.
"""
    
    with open('videos/README.md', 'w') as f:
        f.write(video_readme)
    
    with open('music/README.md', 'w') as f:
        f.write(music_readme)
    
    print("\n" + "="*60)
    print("✅ Setup complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Add your video files to the 'videos/' folder")
    print("2. Add your music files to the 'music/' folder")
    print("   (name them: epic.mp3, calm.mp3, tense.mp3, joyful.mp3, neutral.mp3)")
    print("3. Run: python main.py --input ./videos --music ./music")
    print("\n")

if __name__ == "__main__":
    setup_folders()

