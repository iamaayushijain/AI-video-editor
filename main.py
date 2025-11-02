#!/usr/bin/env python3
"""
Advanced Video Editor with Emotion Detection and Cinematic Effects
"""

import os
import sys
import argparse
import torch
import numpy as np
import cv2
from moviepy.editor import *
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

# Mapping from Kinetics-400 labels to emotional themes
KINETICS_TO_EMOTION_MAP = {
    # Epic / High-Energy
    "surfing water": "epic", "skiing": "epic", "snowboarding": "epic",
    "playing basketball": "epic", "playing american football": "epic",
    "rock climbing": "epic", "running on treadmill": "epic", "dancing": "epic",
    "parkour": "epic", "skydiving": "epic", "driving car": "epic",
    "riding mechanical bull": "epic", "bungee jumping": "epic",
    "playing drums": "epic", "riding mountain bike": "epic",
    
    # Calm / Serene
    "painting": "calm", "reading book": "calm",
    "drinking coffee": "calm", "yoga": "calm", "tai chi": "calm",
    "catching fish": "calm", "sailing": "calm", "sunbathing": "calm",
    "making tea": "calm", "arranging flowers": "calm",
    "meditation": "calm", "walking through forest": "calm",
    
    # Tense / Suspenseful
    "blowing leaves (pile)": "tense", "fencing (sport)": "tense", 
    "archery": "tense", "sword fighting": "tense",
    
    # Joyful / Happy
    "laughing": "joyful", "smiling": "joyful", "hugging": "joyful",
    "playing guitar": "joyful", "celebrating": "joyful",
    "playing with kids": "joyful", "birthday party": "joyful",
    
    # Neutral
    "walking the dog": "neutral", "cooking": "neutral", "shopping": "neutral",
}

DEFAULT_EMOTION = "neutral"

# Music library mapping emotions to music files
MUSIC_LIBRARY = {
    "epic": "epic.mp3",
    "calm": "calm.mp3",
    "tense": "tense.mp3",
    "joyful": "joyful.mp3",
    "neutral": "neutral.mp3"
}

# ==================== CINEMATIC FILTERS ====================

def apply_vintage_filter(frame):
    """Applies a warm, vintage film look"""
    # Slightly reduce saturation
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 0.7  # Reduce saturation
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Add warm tone (sepia-ish)
    b, g, r = cv2.split(frame)
    r = cv2.add(r, 20)
    g = cv2.add(g, 10)
    b = cv2.subtract(b, 15)
    frame = cv2.merge((b, g, r))
    
    # Add slight vignette
    rows, cols = frame.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/2)
    kernel_y = cv2.getGaussianKernel(rows, rows/2)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    mask = np.stack([mask]*3, axis=2)
    frame = (frame * (0.7 + 0.3 * mask)).astype(np.uint8)
    
    return np.clip(frame, 0, 255).astype(np.uint8)

def apply_cool_cinematic_filter(frame):
    """Applies a cool, blue-tinted cinematic look"""
    alpha = 1.15  # Contrast
    beta = -5     # Brightness
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    # Apply cool tint
    b, g, r = cv2.split(adjusted)
    b = cv2.add(b, 20)
    r = cv2.subtract(r, 10)
    g = cv2.add(g, 5)
    final_frame = cv2.merge((b, g, r))
    
    return np.clip(final_frame, 0, 255).astype(np.uint8)

def apply_warm_cinematic_filter(frame):
    """Applies a warm, golden hour look"""
    alpha = 1.1   # Contrast
    beta = 10     # Brightness
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    # Apply warm tint
    b, g, r = cv2.split(adjusted)
    r = cv2.add(r, 25)
    g = cv2.add(g, 15)
    b = cv2.subtract(b, 10)
    final_frame = cv2.merge((b, g, r))
    
    return np.clip(final_frame, 0, 255).astype(np.uint8)

def apply_dramatic_filter(frame):
    """Applies a high-contrast dramatic look"""
    alpha = 1.3   # High contrast
    beta = -10    # Darker shadows
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    # Increase saturation slightly
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)
    final_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return np.clip(final_frame, 0, 255).astype(np.uint8)

def apply_soft_dreamy_filter(frame):
    """Applies a soft, dreamy look with slight blur"""
    # Slight gaussian blur
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # Blend with original
    alpha = 0.7
    frame = cv2.addWeighted(frame, alpha, blurred, 1-alpha, 0)
    
    # Increase brightness slightly
    frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=15)
    
    return np.clip(frame, 0, 255).astype(np.uint8)

def apply_neutral_enhance_filter(frame):
    """Applies subtle enhancement - contrast and sharpness"""
    alpha = 1.1   # Slight contrast
    beta = 5      # Slight brightness
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    # Slight sharpening
    kernel = np.array([[-0.5, -0.5, -0.5],
                       [-0.5,  5.0, -0.5],
                       [-0.5, -0.5, -0.5]])
    sharpened = cv2.filter2D(adjusted, -1, kernel)
    
    # Blend
    final_frame = cv2.addWeighted(adjusted, 0.7, sharpened, 0.3, 0)
    
    return np.clip(final_frame, 0, 255).astype(np.uint8)

# Map emotions to filters
EMOTION_TO_FILTER = {
    "epic": apply_dramatic_filter,
    "calm": apply_soft_dreamy_filter,
    "tense": apply_cool_cinematic_filter,
    "joyful": apply_warm_cinematic_filter,
    "neutral": apply_neutral_enhance_filter
}

# ==================== TRANSITION EFFECTS ====================

def apply_zoom_in_transition(clip, duration=1.0):
    """Applies a zoom-in effect at the start of the clip"""
    def zoom_in(t):
        if t < duration:
            scale = 1 + 0.1 * (1 - t/duration)  # Zoom from 1.1x to 1.0x
            return scale
        return 1
    return clip.resize(lambda t: zoom_in(t))

def apply_zoom_out_transition(clip, duration=1.0):
    """Applies a zoom-out effect at the end of the clip"""
    total_duration = clip.duration
    def zoom_out(t):
        if t > total_duration - duration:
            scale = 1 + 0.1 * ((t - (total_duration - duration))/duration)
            return scale
        return 1
    return clip.resize(lambda t: zoom_out(t))

# ==================== VIDEO ANALYSIS ====================

def load_video_model():
    """Load the VideoMAE model for emotion detection"""
    print("Loading VideoMAE model for emotion detection...")
    try:
        video_feature_extractor = VideoMAEImageProcessor.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics",
            use_auth_token=False
        )
        video_model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics",
            use_auth_token=False
        )
        print("✅ Video model loaded successfully.")
        return video_feature_extractor, video_model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

def get_clip_emotion(video_path, feature_extractor, model):
    """Analyzes a video file and returns its emotional theme"""
    try:
        clip = VideoFileClip(video_path)
        
        # Extract 16 frames evenly spaced
        num_frames = min(16, int(clip.duration * clip.fps))
        frames = []
        for i in range(16):
            t = (i / 16.0) * clip.duration
            frame = clip.get_frame(min(t, clip.duration - 0.1))
            # Convert BGR to RGB if needed
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        clip.close()
        
        inputs = feature_extractor(list(frames), return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_class_idx]
        
        emotion = KINETICS_TO_EMOTION_MAP.get(predicted_label, DEFAULT_EMOTION)
        
        print(f"  - '{os.path.basename(video_path)}': Action='{predicted_label}' → Emotion='{emotion}'")
        return emotion
    except Exception as e:
        print(f"  ⚠️  Error analyzing '{os.path.basename(video_path)}': {e}")
        return DEFAULT_EMOTION

# ==================== VIDEO PROCESSING ====================

def process_video_clip(video_path, clip_index, total_clips, dominant_emotion, clip_duration=4.0):
    """Process a single video clip with appropriate filter and transitions"""
    print(f"  Processing clip {clip_index + 1}/{total_clips}: {os.path.basename(video_path)}")
    
    clip = VideoFileClip(video_path)
    
    # 1. Extract a segment from the middle of the clip
    if clip.duration > clip_duration:
        start_time = max(0, (clip.duration / 2) - (clip_duration / 2))
        clip = clip.subclip(start_time, min(start_time + clip_duration, clip.duration))
    
    # 2. Resize to 720p
    clip = clip.resize(height=720)
    
    # 3. Apply appropriate filter based on emotion
    filter_func = EMOTION_TO_FILTER.get(dominant_emotion, apply_neutral_enhance_filter)
    print(f"    - Applying {filter_func.__name__}")
    clip = clip.fl_image(filter_func)
    
    # 4. Apply transitions based on position in the video
    transition_duration = 1.0
    
    if clip_index == 0:
        # First clip: fade in + zoom in
        print(f"    - Adding opening transition (fade in + zoom in)")
        clip = clip.fx(vfx.fadein, transition_duration)
        clip = apply_zoom_in_transition(clip, transition_duration)
    elif clip_index == total_clips - 1:
        # Last clip: fade out + zoom out
        print(f"    - Adding closing transition (fade out + zoom out)")
        clip = clip.fx(vfx.fadeout, transition_duration)
        clip = apply_zoom_out_transition(clip, transition_duration)
    else:
        # Middle clips: cross-fade
        print(f"    - Adding fade in/out for smooth transitions")
        clip = clip.fx(vfx.fadein, transition_duration).fx(vfx.fadeout, transition_duration)
    
    return clip

def create_cinematic_video(input_folder, music_folder, output_file="cinematic_output.mp4"):
    """Main function to create cinematic video"""
    
    # 1. Get all video files from input folder
    print("\n" + "="*60)
    print("STEP 1: Loading video files")
    print("="*60)
    
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.m4v']
    video_paths = []
    
    for file in sorted(os.listdir(input_folder)):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_paths.append(os.path.join(input_folder, file))
    
    if len(video_paths) < 2:
        print(f"❌ Error: Found only {len(video_paths)} video(s). Need at least 2 videos.")
        sys.exit(1)
    
    print(f"✅ Found {len(video_paths)} video files:")
    for vp in video_paths:
        print(f"   - {os.path.basename(vp)}")
    
    # 2. Analyze emotions in videos
    print("\n" + "="*60)
    print("STEP 2: Analyzing video emotions")
    print("="*60)
    
    feature_extractor, video_model = load_video_model()
    
    clip_emotions = []
    for video_path in video_paths:
        emotion = get_clip_emotion(video_path, feature_extractor, video_model)
        clip_emotions.append(emotion)
    
    # Find dominant emotion
    if clip_emotions:
        emotion_counts = Counter(clip_emotions)
        dominant_emotion = emotion_counts.most_common(1)[0][0]
        print(f"\n✅ Emotion analysis complete!")
        print(f"   Emotion distribution: {dict(emotion_counts)}")
        print(f"   Dominant emotion: '{dominant_emotion.upper()}'")
    else:
        dominant_emotion = DEFAULT_EMOTION
        print(f"\n⚠️  Using default emotion: '{dominant_emotion}'")
    
    # 3. Select appropriate music
    print("\n" + "="*60)
    print("STEP 3: Selecting background music")
    print("="*60)
    
    music_file = MUSIC_LIBRARY.get(dominant_emotion)
    music_path = os.path.join(music_folder, music_file) if music_file else None
    
    main_audio = None
    if music_path and os.path.exists(music_path):
        print(f"✅ Selected music: '{music_file}' for '{dominant_emotion}' theme")
        main_audio = AudioFileClip(music_path)
    else:
        # Try to find any music file as fallback
        for file in os.listdir(music_folder):
            if file.lower().endswith(('.mp3', '.wav', '.m4a')):
                fallback_path = os.path.join(music_folder, file)
                print(f"⚠️  Music for '{dominant_emotion}' not found. Using fallback: '{file}'")
                main_audio = AudioFileClip(fallback_path)
                break
    
    if main_audio is None:
        print("⚠️  No music files found. Video will have no background music.")
    
    # 4. Process all video clips
    print("\n" + "="*60)
    print("STEP 4: Processing video clips")
    print("="*60)
    
    processed_clips = []
    clip_duration = 4.0  # Standard clip length
    
    for i, video_path in enumerate(video_paths):
        try:
            processed_clip = process_video_clip(
                video_path, i, len(video_paths), 
                dominant_emotion, clip_duration
            )
            processed_clips.append(processed_clip)
        except Exception as e:
            print(f"  ❌ Error processing '{os.path.basename(video_path)}': {e}")
            continue
    
    if not processed_clips:
        print("❌ No clips were successfully processed!")
        sys.exit(1)
    
    print(f"✅ Successfully processed {len(processed_clips)} clips")
    
    # 5. Concatenate clips
    print("\n" + "="*60)
    print("STEP 5: Assembling final video")
    print("="*60)
    
    print("Concatenating clips...")
    final_video = concatenate_videoclips(processed_clips, method="compose")
    
    # 6. Add music
    if main_audio:
        print("Adding background music...")
        # Loop audio to match video duration
        audio_duration = final_video.duration
        main_audio = main_audio.fx(afx.audio_loop, duration=audio_duration)
        
        # Reduce volume slightly and fade out at the end
        main_audio = main_audio.fx(afx.volumex, 0.7)
        main_audio = main_audio.fx(afx.audio_fadeout, 2.0)
        
        final_video = final_video.set_audio(main_audio)
    
    print(f"✅ Assembly complete! Total duration: {final_video.duration:.2f} seconds")
    
    # 7. Write output file
    print("\n" + "="*60)
    print("STEP 6: Rendering final video")
    print("="*60)
    print(f"Output file: {output_file}")
    print("This may take several minutes depending on video length...")
    
    try:
        final_video.write_videofile(
            output_file,
            codec='libx264',
            audio_codec='aac',
            fps=24,
            preset='medium',
            bitrate='5000k',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            logger='bar'
        )
        
        print("\n" + "="*60)
        print("🎉 SUCCESS! Your cinematic video is ready! 🎉")
        print("="*60)
        print(f"Output file: {os.path.abspath(output_file)}")
        print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        
        # Clean up
        for clip in processed_clips:
            clip.close()
        final_video.close()
        if main_audio:
            main_audio.close()
        
    except Exception as e:
        print(f"\n❌ Error during video rendering: {e}")
        sys.exit(1)

# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(
        description='Create cinematic videos with AI-powered emotion detection and effects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input ./videos
  python main.py --input ./videos --output my_video.mp4
  python main.py --input ./videos --music /path/to/custom/music
        """
    )
    
    parser.add_argument('--input', '-i', 
                       required=True,
                       help='Path to folder containing input video files')
    
    parser.add_argument('--music', '-m',
                       default='./music',
                       help='Path to folder containing music files (default: ./music)')
    
    parser.add_argument('--output', '-o',
                       default='cinematic_output.mp4',
                       help='Output video filename (default: cinematic_output.mp4)')
    
    args = parser.parse_args()
    
    # Validate input folder
    if not os.path.isdir(args.input):
        print(f"❌ Error: Input folder '{args.input}' does not exist!")
        sys.exit(1)
    
    # Validate music folder
    if not os.path.isdir(args.music):
        print(f"❌ Error: Music folder '{args.music}' does not exist!")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("CINEMATIC VIDEO EDITOR")
    print("="*60)
    print(f"Input folder: {os.path.abspath(args.input)}")
    print(f"Music folder: {os.path.abspath(args.music)}")
    print(f"Output file: {args.output}")
    
    create_cinematic_video(args.input, args.music, args.output)

if __name__ == "__main__":
    main()
