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
import time
from datetime import timedelta
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

# ==================== FRAME QUALITY ANALYSIS ====================

def calculate_frame_sharpness(frame):
    """Calculate sharpness using Laplacian variance"""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def calculate_frame_vibrancy(frame):
    """Calculate color vibrancy (saturation)"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    return np.mean(saturation)

def calculate_frame_brightness(frame):
    """Calculate brightness and check if well-exposed"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    value = hsv[:, :, 2]
    mean_brightness = np.mean(value)
    # Penalize over/under exposure
    optimal_brightness = 128
    brightness_score = 100 - abs(mean_brightness - optimal_brightness)
    return brightness_score

def calculate_frame_contrast(frame):
    """Calculate contrast using standard deviation"""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return np.std(gray)

def calculate_frame_quality_score(frame):
    """Calculate overall frame quality score"""
    sharpness = calculate_frame_sharpness(frame)
    vibrancy = calculate_frame_vibrancy(frame)
    brightness = calculate_frame_brightness(frame)
    contrast = calculate_frame_contrast(frame)
    
    # Normalize and weight the scores
    # Sharpness is most important for quality
    quality_score = (
        sharpness * 0.4 +  # Weight sharpness highly
        vibrancy * 0.3 +   # Vibrant colors are appealing
        brightness * 0.15 + # Good exposure
        contrast * 0.15     # Good contrast
    )
    
    return quality_score

def select_best_frames(frames, num_frames=16):
    """Select the best quality frames from a larger set"""
    frame_scores = []
    
    for i, frame in enumerate(frames):
        score = calculate_frame_quality_score(frame)
        frame_scores.append((i, score, frame))
    
    # Sort by score and take top num_frames
    frame_scores.sort(key=lambda x: x[1], reverse=True)
    best_frames = [frame for _, _, frame in frame_scores[:num_frames]]
    
    return best_frames

def detect_scene_changes(video_path, threshold=30.0):
    """Detect scene changes in video using histogram difference"""
    cap = cv2.VideoCapture(video_path)
    
    scene_timestamps = [0.0]  # Start is always a scene
    prev_hist = None
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate histogram for current frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        if prev_hist is not None:
            # Compare with previous frame
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)
            
            if diff > threshold:
                timestamp = frame_count / fps
                # Avoid detecting changes too close together
                if timestamp - scene_timestamps[-1] > 2.0:
                    scene_timestamps.append(timestamp)
        
        prev_hist = hist
        frame_count += 1
    
    cap.release()
    return scene_timestamps

# ==================== TRANSITION EFFECTS ====================

def apply_zoom_in_transition(clip, duration=1.0):
    """Applies a zoom-in effect at the start of the clip"""
    def zoom_in(t):
        if t < duration:
            scale = 1 + 0.15 * (1 - t/duration)  # Zoom from 1.15x to 1.0x
            return scale
        return 1
    return clip.resize(lambda t: zoom_in(t))

def apply_zoom_out_transition(clip, duration=1.0):
    """Applies a zoom-out effect at the end of the clip"""
    total_duration = clip.duration
    def zoom_out(t):
        if t > total_duration - duration:
            scale = 1 + 0.15 * ((t - (total_duration - duration))/duration)
            return scale
        return 1
    return clip.resize(lambda t: zoom_out(t))

def apply_slide_in_transition(clip, duration=1.0):
    """Applies a slide-in effect from the right"""
    w, h = clip.size
    def slide_in(t):
        if t < duration:
            offset = int(w * (1 - t/duration))
            return (offset, 0)
        return (0, 0)
    return clip.set_position(lambda t: slide_in(t))

def apply_dramatic_zoom_rotate(clip, duration=1.5):
    """Applies dramatic zoom with slight rotation for video transitions"""
    def zoom_rotate(t):
        if t < duration:
            progress = t / duration
            scale = 1 + 0.2 * progress  # Zoom from 1.0x to 1.2x
            return scale
        return 1.2
    return clip.resize(lambda t: zoom_rotate(t))

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
    """Analyzes a video file and returns its emotional theme using intelligent frame selection"""
    try:
        clip = VideoFileClip(video_path)
        
        # Extract 32 frames evenly spaced (more samples for better quality selection)
        num_candidate_frames = 32
        candidate_frames = []
        
        for i in range(num_candidate_frames):
            t = (i / float(num_candidate_frames)) * clip.duration
            frame = clip.get_frame(min(t, clip.duration - 0.1))
            # Convert BGR to RGB if needed
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            candidate_frames.append(frame)
        
        clip.close()
        
        # Select the best 16 frames based on quality metrics
        print(f"    - Analyzing frame quality...")
        best_frames = select_best_frames(candidate_frames, num_frames=16)
        
        inputs = feature_extractor(list(best_frames), return_tensors="pt")
        
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

def extract_best_segments(video_path, min_duration=3.0, max_segments=3):
    """Extract multiple best quality segments from a video based on scene detection"""
    print(f"    - Detecting scenes in video...")
    
    # Detect scene changes
    scene_timestamps = detect_scene_changes(video_path)
    
    clip = VideoFileClip(video_path)
    total_duration = clip.duration
    
    # Create segments based on scene changes
    segments = []
    for i in range(len(scene_timestamps)):
        start = scene_timestamps[i]
        end = scene_timestamps[i + 1] if i + 1 < len(scene_timestamps) else total_duration
        
        # Only keep segments that are long enough
        if end - start >= min_duration:
            # Score this segment by analyzing frames
            segment_quality = analyze_segment_quality(clip, start, min(start + 5, end))
            segments.append({
                'start': start,
                'end': end,
                'duration': end - start,
                'quality': segment_quality
            })
    
    clip.close()
    
    # If no good segments found, use the whole video
    if not segments:
        segments = [{
            'start': 0,
            'end': min(total_duration, 5.0),
            'duration': min(total_duration, 5.0),
            'quality': 50.0
        }]
    
    # Sort by quality and take top segments
    segments.sort(key=lambda x: x['quality'], reverse=True)
    best_segments = segments[:max_segments]
    
    # Sort by start time to maintain chronological order
    best_segments.sort(key=lambda x: x['start'])
    
    print(f"    - Found {len(scene_timestamps)} scenes, selected {len(best_segments)} best segments")
    
    return best_segments

def analyze_segment_quality(clip, start_time, end_time, num_samples=5):
    """Analyze quality of a video segment"""
    quality_scores = []
    
    for i in range(num_samples):
        t = start_time + (i / float(num_samples)) * (end_time - start_time)
        if t < clip.duration:
            frame = clip.get_frame(t)
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            quality = calculate_frame_quality_score(frame)
            quality_scores.append(quality)
    
    return np.mean(quality_scores) if quality_scores else 0.0

def process_video_segment(video_path, segment_info, source_index, segment_index, dominant_emotion, clip_duration=4.0):
    """Process a single video segment with appropriate filter"""
    print(f"    - Processing segment {segment_index} from {os.path.basename(video_path)}")
    
    clip = VideoFileClip(video_path)
    
    # Extract the segment
    start = segment_info['start']
    duration = min(segment_info['duration'], clip_duration)
    end = start + duration
    
    clip = clip.subclip(start, min(end, clip.duration))
    
    # Resize to 720p
    clip = clip.resize(height=720)
    
    # Apply appropriate filter based on emotion
    filter_func = EMOTION_TO_FILTER.get(dominant_emotion, apply_neutral_enhance_filter)
    print(f"      - Applying {filter_func.__name__}")
    clip = clip.fl_image(filter_func)
    
    # Return the processed clip with metadata
    return {
        'clip': clip,
        'source_index': source_index,
        'segment_index': segment_index,
        'video_path': video_path
    }

def apply_adaptive_transition(clip_data, prev_clip_data, is_first, is_last, transition_duration=1.0):
    """Apply adaptive transitions based on whether clips are from same source"""
    clip = clip_data['clip']
    
    if is_first:
        # First clip: dramatic opening with fade in + zoom in
        print(f"      - Applying DRAMATIC opening transition")
        clip = clip.fx(vfx.fadein, transition_duration)
        clip = apply_zoom_in_transition(clip, transition_duration)
    elif is_last:
        # Last clip: dramatic closing with fade out + zoom out
        print(f"      - Applying DRAMATIC closing transition")
        clip = clip.fx(vfx.fadeout, transition_duration)
        clip = apply_zoom_out_transition(clip, transition_duration)
    else:
        # Check if same source as previous clip
        if prev_clip_data and clip_data['source_index'] == prev_clip_data['source_index']:
            # Same source video: subtle transition (just crossfade)
            print(f"      - Applying SUBTLE transition (same source)")
            clip = clip.fx(vfx.fadein, 0.5).fx(vfx.fadeout, 0.5)
        else:
            # Different source video: dramatic transition
            print(f"      - Applying DRAMATIC transition (new source)")
            clip = clip.fx(vfx.fadein, transition_duration).fx(vfx.fadeout, transition_duration)
            # Add slight zoom for impact
            clip = apply_zoom_in_transition(clip, transition_duration * 0.7)
    
    clip_data['clip'] = clip
    return clip_data

def create_cinematic_video(input_folder, music_folder, output_file="cinematic_output.mp4"):
    """Main function to create cinematic video with intelligent editing"""
    
    start_time = time.time()
    step_times = {}
    
    # 1. Get all video files from input folder
    step_start = time.time()
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
    
    step_times['load_files'] = time.time() - step_start
    
    # 2. Analyze emotions in videos
    step_start = time.time()
    print("\n" + "="*60)
    print("STEP 2: Analyzing video emotions (AI-powered)")
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
    
    step_times['emotion_analysis'] = time.time() - step_start
    
    # 3. Extract best segments from videos
    step_start = time.time()
    print("\n" + "="*60)
    print("STEP 3: Intelligent scene detection & segment extraction")
    print("="*60)
    
    all_segments = []
    for i, video_path in enumerate(video_paths):
        print(f"\n  Analyzing {os.path.basename(video_path)}...")
        try:
            segments = extract_best_segments(video_path, min_duration=3.0, max_segments=2)
            for seg in segments:
                all_segments.append({
                    'video_path': video_path,
                    'segment_info': seg,
                    'source_index': i
                })
        except Exception as e:
            print(f"  ⚠️  Error extracting segments from '{os.path.basename(video_path)}': {e}")
    
    print(f"\n✅ Extracted {len(all_segments)} high-quality segments from {len(video_paths)} videos")
    step_times['scene_detection'] = time.time() - step_start
    
    # 4. Select appropriate music
    step_start = time.time()
    print("\n" + "="*60)
    print("STEP 4: Selecting background music")
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
    
    step_times['music_selection'] = time.time() - step_start
    
    # 5. Process all video segments
    step_start = time.time()
    print("\n" + "="*60)
    print("STEP 5: Processing segments with cinematic filters")
    print("="*60)
    
    processed_clips = []
    clip_duration = 3.0
    
    for i, seg_data in enumerate(all_segments):
        try:
            processed = process_video_segment(
                seg_data['video_path'],
                seg_data['segment_info'],
                seg_data['source_index'],
                i,
                dominant_emotion,
                clip_duration
            )
            processed_clips.append(processed)
        except Exception as e:
            print(f"  ❌ Error processing segment: {e}")
            continue
    
    if not processed_clips:
        print("❌ No clips were successfully processed!")
        sys.exit(1)
    
    print(f"\n✅ Successfully processed {len(processed_clips)} segments")
    step_times['filter_processing'] = time.time() - step_start
    
    # 6. Apply adaptive transitions
    step_start = time.time()
    print("\n" + "="*60)
    print("STEP 6: Applying adaptive transitions")
    print("="*60)
    print("  (Subtle transitions within same video, dramatic between different videos)")
    
    for i in range(len(processed_clips)):
        is_first = (i == 0)
        is_last = (i == len(processed_clips) - 1)
        prev_clip = processed_clips[i-1] if i > 0 else None
        
        processed_clips[i] = apply_adaptive_transition(
            processed_clips[i],
            prev_clip,
            is_first,
            is_last
        )
    
    step_times['transitions'] = time.time() - step_start
    
    # 7. Concatenate clips
    step_start = time.time()
    print("\n" + "="*60)
    print("STEP 7: Assembling final video")
    print("="*60)
    
    print("Concatenating all segments...")
    clips_only = [c['clip'] for c in processed_clips]
    final_video = concatenate_videoclips(clips_only, method="compose")
    
    # 8. Add music
    if main_audio:
        print("Adding background music with dynamic volume...")
        audio_duration = final_video.duration
        main_audio = main_audio.fx(afx.audio_loop, duration=audio_duration)
        main_audio = main_audio.fx(afx.volumex, 0.7)
        main_audio = main_audio.fx(afx.audio_fadeout, 2.0)
        final_video = final_video.set_audio(main_audio)
    
    print(f"✅ Assembly complete! Total duration: {final_video.duration:.2f} seconds")
    step_times['assembly'] = time.time() - step_start
    
    # 9. Write output file
    step_start = time.time()
    print("\n" + "="*60)
    print("STEP 8: Rendering final video")
    print("="*60)
    print(f"Output file: {output_file}")
    print("Encoding with H.264... This may take several minutes...")
    
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
        
        step_times['rendering'] = time.time() - step_start
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("🎉 SUCCESS! Your cinematic video is ready! 🎉")
        print("="*60)
        print(f"Output file: {os.path.abspath(output_file)}")
        print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        print(f"Video duration: {final_video.duration:.2f} seconds")
        
        # Display timing breakdown
        print("\n" + "="*60)
        print("⏱️  PERFORMANCE BREAKDOWN")
        print("="*60)
        print(f"  Load files:           {str(timedelta(seconds=int(step_times['load_files'])))}")
        print(f"  AI emotion analysis:  {str(timedelta(seconds=int(step_times['emotion_analysis'])))}")
        print(f"  Scene detection:      {str(timedelta(seconds=int(step_times['scene_detection'])))}")
        print(f"  Music selection:      {str(timedelta(seconds=int(step_times['music_selection'])))}")
        print(f"  Apply filters:        {str(timedelta(seconds=int(step_times['filter_processing'])))}")
        print(f"  Apply transitions:    {str(timedelta(seconds=int(step_times['transitions'])))}")
        print(f"  Assembly:             {str(timedelta(seconds=int(step_times['assembly'])))}")
        print(f"  Video rendering:      {str(timedelta(seconds=int(step_times['rendering'])))}")
        print(f"  " + "-"*40)
        print(f"  TOTAL TIME:           {str(timedelta(seconds=int(total_time)))}")
        print("="*60)
        
        # Clean up
        for clip_data in processed_clips:
            clip_data['clip'].close()
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
