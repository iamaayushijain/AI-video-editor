#!/usr/bin/env python3
"""
Cinematic Video Editor with AI-Powered Emotion Detection
Automatically creates polished videos with smart transitions and effects.
"""

import os
import sys
import argparse
import time
import random
import warnings
from datetime import timedelta
from collections import Counter

import numpy as np
import cv2
import torch
from moviepy.editor import *
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

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
    kernel_x = cv2.getGaussianKernel(cols, cols//2)
    kernel_y = cv2.getGaussianKernel(rows, rows//2)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    mask = np.stack([mask]*3, axis=2)
    frame = (frame * (0.7 + 0.3 * mask)).astype(np.uint8)
    
    return np.clip(frame, 0, 255).astype(np.uint8)

def apply_cool_cinematic_filter(frame):
    """Applies a cool, blue-tinted cinematic look"""
    alpha = 1.25
    beta = -10
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    # Apply cool tint
    b, g, r = cv2.split(adjusted)
    b = cv2.add(b, 30)
    r = cv2.subtract(r, 15)
    g = cv2.add(g, 10)
    final_frame = cv2.merge((b, g, r))
    
    # Add slight desaturation for moody look
    hsv = cv2.cvtColor(final_frame, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.9, 0, 255)
    final_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return np.clip(final_frame, 0, 255).astype(np.uint8)

def apply_warm_cinematic_filter(frame):
    """Applies a warm, golden hour look"""
    alpha = 1.2
    beta = 15
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    # Apply warm tint
    b, g, r = cv2.split(adjusted)
    r = cv2.add(r, 35)
    g = cv2.add(g, 20)
    b = cv2.subtract(b, 15)
    final_frame = cv2.merge((b, g, r))
    
    # Boost saturation for vibrant look
    hsv = cv2.cvtColor(final_frame, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
    final_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return np.clip(final_frame, 0, 255).astype(np.uint8)

def apply_dramatic_filter(frame):
    """Applies a high-contrast dramatic look"""
    alpha = 1.5
    beta = -15
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    # Increase saturation
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
    final_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Add strong vignette for drama
    h, w = final_frame.shape[:2]
    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    radius = np.sqrt(X**2 + Y**2)
    vignette = 1 - np.clip(radius * 0.5, 0, 0.4)
    vignette = np.stack([vignette]*3, axis=2)
    final_frame = (final_frame * vignette).astype(np.uint8)
    
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
    
    # Weighted quality score
    quality_score = (
        sharpness * 0.4 +
        vibrancy * 0.3 +
        brightness * 0.15 +
        contrast * 0.15
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

# ==================== VISUAL EFFECTS & OVERLAYS ====================

def create_heart_particle(frame, x, y, size, color=(255, 20, 147), alpha=0.8):
    """Draw a proper heart shape using mathematical curve"""
    # Ensure frame is uint8
    frame = frame.astype(np.uint8)
    h, w = frame.shape[:2]
    
    # Check bounds
    if x < 0 or x >= w or y < 0 or y >= h:
        return frame
    
    # Create heart shape using parametric equations
    heart_points = []
    for t in np.linspace(0, 2*np.pi, 100):
        # Parametric heart curve equations
        heart_x = 16 * np.sin(t)**3
        heart_y = -(13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
        
        # Scale and translate
        px = int(x + heart_x * size / 20)
        py = int(y + heart_y * size / 20)
        
        if 0 <= px < w and 0 <= py < h:
            heart_points.append([px, py])
    
    if len(heart_points) > 3:
        heart_points = np.array(heart_points, dtype=np.int32)
        
        # Create a mask for the heart
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [heart_points], 255)
        
        # Apply the heart with alpha blending
        for c in range(3):
            frame[:, :, c] = np.where(
                mask == 255,
                frame[:, :, c] * (1 - alpha) + color[c] * alpha,
                frame[:, :, c]
            ).astype(np.uint8)
    
    return frame

def create_sparkle(frame, x, y, size, intensity=255):
    """Draw a sparkle/star at given position"""
    frame = frame.astype(np.uint8)
    color = (int(intensity), int(intensity), int(intensity * 0.8))
    
    # Ensure coordinates are within bounds
    h, w = frame.shape[:2]
    if x < size or x > w-size or y < size or y > h-size:
        return frame
    
    # Draw cross pattern for sparkle
    cv2.line(frame, (x-size, y), (x+size, y), color, 2)
    cv2.line(frame, (x, y-size), (x, y+size), color, 2)
    cv2.line(frame, (x-size//2, y-size//2), (x+size//2, y+size//2), color, 1)
    cv2.line(frame, (x-size//2, y+size//2), (x+size//2, y-size//2), color, 1)
    
    return frame

def apply_celebration_effect(frame, t, duration):
    """Add celebration confetti and sparkles"""
    if t > duration:
        return frame
    
    frame = frame.astype(np.uint8)
    progress = t / duration
    h, w = frame.shape[:2]
    
    # Random confetti particles
    num_particles = int(50 * (1 - progress))
    for _ in range(num_particles):
        x = random.randint(0, w-1)
        y = int(random.random() * h * progress)  # Fall down
        color = tuple([random.randint(100, 255) for _ in range(3)])
        size = random.randint(4, 12)  # Bigger particles
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame, (x, y), size, color, -1)
    
    # Sparkles
    num_sparkles = int(25 * (1 - progress))
    if w > 100 and h > 100:  # Only if frame is big enough
        for _ in range(num_sparkles):
            x = random.randint(50, w-50)
            y = random.randint(50, h-50)
            size = random.randint(8, 20)  # Bigger sparkles
            intensity = random.randint(200, 255)
            frame = create_sparkle(frame, x, y, size, intensity)
    
    return frame

def apply_hearts_effect(frame, t, duration):
    """Add floating hearts effect"""
    if t > duration:
        return frame
    
    progress = t / duration
    h, w = frame.shape[:2]
    
    # Floating hearts
    num_hearts = int(20 * (1 - progress**2))
    for i in range(num_hearts):
        seed = i * 1000 + int(t * 100)
        random.seed(seed)
        x = random.randint(0, w-100)
        y_base = random.randint(0, h)
        y = int(y_base - progress * h * 0.4)  # Float higher
        size = random.randint(20, 40)  # Bigger hearts
        color = [(255, 20, 147), (255, 105, 180), (255, 192, 203), (255, 0, 100)][i % 4]
        alpha = max(0.4, 0.9 * (1 - progress))  # More visible
        frame = create_heart_particle(frame, x, y, size, color, alpha)
    
    random.seed()  # Reset seed
    return frame

def apply_light_leak_effect(frame, color=(255, 200, 150), intensity=0.3):
    """Add cinematic light leak effect"""
    h, w = frame.shape[:2]
    
    # Create gradient overlay
    overlay = np.zeros_like(frame, dtype=np.float32)
    
    # Diagonal light leak
    for i in range(h):
        for j in range(w):
            dist = np.sqrt((i/h)**2 + (j/w)**2)
            if dist < 0.7:
                factor = (0.7 - dist) / 0.7
                overlay[i, j] = [c * factor * intensity for c in color]
    
    frame = cv2.addWeighted(frame, 1, overlay.astype(np.uint8), 1, 0)
    return np.clip(frame, 0, 255).astype(np.uint8)

def apply_film_grain(frame, intensity=0.1):
    """Add film grain for vintage effect"""
    noise = np.random.normal(0, 25*intensity, frame.shape).astype(np.int16)
    frame = frame.astype(np.int16) + noise
    return np.clip(frame, 0, 255).astype(np.uint8)

def create_vignette_strong(frame, strength=0.6):
    """Create strong vignette effect"""
    h, w = frame.shape[:2]
    
    # Create radial gradient
    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    radius = np.sqrt(X**2 + Y**2)
    vignette = 1 - np.clip(radius * strength, 0, 1)
    vignette = np.stack([vignette]*3, axis=2)
    
    frame = (frame * vignette).astype(np.uint8)
    return frame

def apply_screen_flash(frame, intensity=1.0):
    """Apply white screen flash effect"""
    flash_overlay = np.ones_like(frame) * 255
    return cv2.addWeighted(frame.astype(np.uint8), 1-intensity, 
                           flash_overlay.astype(np.uint8), intensity, 0)

def apply_color_burst(frame, t, duration, color='rainbow'):
    """Apply colorful burst effect"""
    if t > duration:
        return frame
    
    progress = t / duration
    intensity = (1 - progress) * 0.5  # Fade out
    
    h, w = frame.shape[:2]
    
    if color == 'rainbow':
        # Create rainbow gradient
        overlay = np.zeros_like(frame)
        for i in range(h):
            hue_val = int(180 * (i / h))
            hsv_color = np.uint8([[[hue_val, 255, 255]]])
            rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]
            overlay[i, :] = rgb_color
    elif color == 'gold':
        overlay = np.full_like(frame, (50, 215, 255))  # Gold color
    elif color == 'purple':
        overlay = np.full_like(frame, (128, 0, 255))  # Purple
    else:
        overlay = np.full_like(frame, (255, 255, 255))  # White
    
    return cv2.addWeighted(frame.astype(np.uint8), 1, 
                           overlay.astype(np.uint8), intensity, 0)

# ==================== TRANSITION EFFECTS ====================

def apply_zoom_in_transition(clip, duration=1.0, intensity='medium'):
    """Applies a dramatic zoom-in effect"""
    zoom_amount = {'low': 0.15, 'medium': 0.3, 'high': 0.5, 'extreme': 0.8}
    zoom = zoom_amount.get(intensity, 0.3)
    
    def zoom_in(t):
        if t < duration:
            scale = 1 + zoom * (1 - t/duration)  # More dramatic zoom
            return scale
        return 1
    return clip.resize(lambda t: zoom_in(t))

def apply_zoom_out_transition(clip, duration=1.0, intensity='medium'):
    """Applies a dramatic zoom-out effect"""
    zoom_amount = {'low': 0.15, 'medium': 0.3, 'high': 0.5, 'extreme': 0.8}
    zoom = zoom_amount.get(intensity, 0.3)
    
    total_duration = clip.duration
    def zoom_out(t):
        if t > total_duration - duration:
            scale = 1 + zoom * ((t - (total_duration - duration))/duration)
            return scale
        return 1
    return clip.resize(lambda t: zoom_out(t))

def apply_scale_in_transition(clip, duration=1.0):
    """Applies a dramatic scale-in effect"""
    def scale_in(t):
        if t < duration:
            scale = 0.5 + 0.5 * (t/duration)
            return scale
        return 1
    return clip.resize(lambda t: scale_in(t))

def apply_explosive_zoom(clip, duration=1.5):
    """Explosive zoom in effect"""
    def explosive(t):
        if t < duration:
            progress = t / duration
            # Exponential zoom for explosive effect
            scale = 1 + 1.0 * (1 - progress)**2
            return scale
        return 1
    return clip.resize(lambda t: explosive(t))

def apply_slide_in_left(clip, duration=1.0):
    """Slide in from left"""
    w, h = clip.size
    def slide(t):
        if t < duration:
            x = -w * (1 - t/duration)
            return (x, 'center')
        return ('center', 'center')
    return clip.set_position(lambda t: slide(t))

def apply_slide_in_right(clip, duration=1.0):
    """Slide in from right"""
    w, h = clip.size
    def slide(t):
        if t < duration:
            x = w * (1 - t/duration)
            return (x, 'center')
        return ('center', 'center')
    return clip.set_position(lambda t: slide(t))

def apply_slide_out_left(clip, duration=1.0):
    """Slide out to left"""
    w, h = clip.size
    total_duration = clip.duration
    def slide(t):
        if t > total_duration - duration:
            progress = (t - (total_duration - duration)) / duration
            x = -w * progress
            return (x, 'center')
        return ('center', 'center')
    return clip.set_position(lambda t: slide(t))

def apply_slide_out_right(clip, duration=1.0):
    """Slide out to right"""
    w, h = clip.size
    total_duration = clip.duration
    def slide(t):
        if t > total_duration - duration:
            progress = (t - (total_duration - duration)) / duration
            x = w * progress
            return (x, 'center')
        return ('center', 'center')
    return clip.set_position(lambda t: slide(t))

def apply_ken_burns_effect(clip, duration=1.0, zoom_direction='in'):
    """Apply Ken Burns pan and zoom effect"""
    if zoom_direction == 'in':
        # Zoom in effect
        return apply_zoom_in_transition(clip, duration)
    else:
        # Zoom out effect
        return apply_zoom_out_transition(clip, duration)

def apply_wipe_transition(clip, duration=1.0, direction='horizontal'):
    """Apply wipe transition"""
    w, h = clip.size
    
    def mask_fn(t):
        if t < duration:
            progress = t / duration
            mask = np.zeros((h, w), dtype=np.uint8)
            if direction == 'horizontal':
                cutoff = int(w * progress)
                mask[:, :cutoff] = 255
            else:  # vertical
                cutoff = int(h * progress)
                mask[:cutoff, :] = 255
            return mask
        return np.ones((h, w), dtype=np.uint8) * 255
    
    # Note: MoviePy doesn't support custom masks easily, so we'll use fade + slide combo
    if direction == 'horizontal':
        return clip.fx(vfx.fadein, duration)
    else:
        return clip.fx(vfx.fadein, duration)

# Transition types list for variety
TRANSITION_TYPES = [
    'zoom_in', 'zoom_out', 'slide_left', 'slide_right', 
    'spin', 'rotate', 'fade', 'crossfade'
]

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
    
    # Resize to 1080p (Full HD) for better quality
    clip = clip.resize(height=1080)
    
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

def apply_diverse_transitions(clip_data, prev_clip_data, clip_index, total_clips, dominant_emotion, transition_duration=1.0):
    """Apply diverse transitions with special effects"""
    clip = clip_data['clip']
    is_first = (clip_index == 0)
    is_last = (clip_index == total_clips - 1)
    
    if is_first:
        # First clip: Explosive opening
        print(f"      - Applying explosive opening (flash + confetti + zoom)")
        
        # Add WHITE FLASH at the very start + celebration
        def explosive_opening(get_frame, t):
            frame = get_frame(t)
            # Intense white flash in first 0.2 seconds
            if t < 0.2:
                flash_intensity = 0.8 * (1 - t/0.2)
                frame = apply_screen_flash(frame.copy(), flash_intensity)
            # Confetti and sparkles
            if t < 3.0:
                frame = apply_celebration_effect(frame.copy(), t, 3.0)
            # Color burst for extra flair
            if t < 1.5:
                frame = apply_color_burst(frame.copy(), t, 1.5, 'rainbow')
            return frame
        
        clip = clip.fl(explosive_opening)
        clip = clip.fx(vfx.fadein, transition_duration * 0.5)  # Quick fade
        clip = apply_explosive_zoom(clip, transition_duration * 1.5)  # Explosive zoom
        
    elif is_last:
        # Last clip: Dramatic finale
        print(f"      - Applying dramatic finale (hearts + glow + zoom)")
        
        # Add hearts + glow effect for emotional ending
        def dramatic_finale(get_frame, t):
            frame = get_frame(t)
            remaining = clip.duration - t
            # Hearts throughout the ending
            if remaining < 3.0:
                frame = apply_hearts_effect(frame.copy(), 3.0 - remaining, 3.0)
            # Bright glow at the very end
            if remaining < 0.5:
                glow_intensity = (0.5 - remaining) / 0.5 * 0.3
                frame = apply_screen_flash(frame.copy(), glow_intensity)
            return frame
        
        clip = clip.fl(dramatic_finale)
        clip = clip.fx(vfx.fadeout, transition_duration * 1.5)  # Slow fade
        clip = apply_zoom_out_transition(clip, transition_duration * 1.5, 'high')  # Dramatic zoom
        
    else:
        # MIDDLE CLIPS: Super varied and dramatic
        same_source = prev_clip_data and clip_data['source_index'] == prev_clip_data['source_index']
        
        if same_source:
            # Dynamic transitions within same video
            transitions = ['zoom_medium', 'scale_dramatic', 'fade_zoom']
            transition_type = random.choice(transitions)
            
            print(f"      - Applying DYNAMIC {transition_type} (same source)")
            
            clip = clip.fx(vfx.fadein, 0.7).fx(vfx.fadeout, 0.7)
            
            if transition_type == 'zoom_medium':
                clip = apply_zoom_in_transition(clip, 0.8, 'medium')
            elif transition_type == 'scale_dramatic':
                clip = apply_scale_in_transition(clip, 0.9)
            elif transition_type == 'fade_zoom':
                clip = apply_zoom_in_transition(clip, 0.7, 'low')
                
        else:
            # Dramatic transitions between different videos
            dramatic_transitions = ['explosive', 'slide_flash_left', 'slide_flash_right', 'zoom_burst', 'scale_flash']
            transition_type = random.choice(dramatic_transitions)
            
            print(f"      - Applying dramatic {transition_type} (new source)")
            
            if transition_type == 'explosive':
                # Flash + explosive zoom
                def flash_entry(get_frame, t):
                    frame = get_frame(t)
                    if t < 0.15:
                        flash_intensity = 0.6 * (1 - t/0.15)
                        frame = apply_screen_flash(frame.copy(), flash_intensity)
                    return frame
                clip = clip.fl(flash_entry)
                clip = clip.fx(vfx.fadein, transition_duration * 0.5)
                clip = apply_explosive_zoom(clip, transition_duration * 1.2)
                clip = clip.fx(vfx.fadeout, transition_duration)
                
            elif transition_type == 'slide_flash_left':
                # Slide + flash
                def flash_entry(get_frame, t):
                    frame = get_frame(t)
                    if t < 0.1:
                        frame = apply_screen_flash(frame.copy(), 0.5)
                    return frame
                clip = clip.fl(flash_entry)
                clip = clip.fx(vfx.fadein, transition_duration * 0.8)
                clip = apply_slide_in_left(clip, transition_duration)
                clip = clip.fx(vfx.fadeout, transition_duration)
                
            elif transition_type == 'slide_flash_right':
                # Slide + flash
                def flash_entry(get_frame, t):
                    frame = get_frame(t)
                    if t < 0.1:
                        frame = apply_screen_flash(frame.copy(), 0.5)
                    return frame
                clip = clip.fl(flash_entry)
                clip = clip.fx(vfx.fadein, transition_duration * 0.8)
                clip = apply_slide_in_right(clip, transition_duration)
                clip = clip.fx(vfx.fadeout, transition_duration)
                
            elif transition_type == 'zoom_burst':
                # Color burst + zoom
                def burst_entry(get_frame, t):
                    frame = get_frame(t)
                    if t < 1.0:
                        frame = apply_color_burst(frame.copy(), t, 1.0, 'gold')
                    return frame
                clip = clip.fl(burst_entry)
                clip = clip.fx(vfx.fadein, transition_duration)
                clip = apply_zoom_in_transition(clip, transition_duration, 'high')
                clip = clip.fx(vfx.fadeout, transition_duration)
                
            elif transition_type == 'scale_flash':
                # Flash + dramatic scale
                def flash_entry(get_frame, t):
                    frame = get_frame(t)
                    if t < 0.1:
                        frame = apply_screen_flash(frame.copy(), 0.6)
                    return frame
                clip = clip.fl(flash_entry)
                clip = clip.fx(vfx.fadein, transition_duration * 0.6)
                clip = apply_scale_in_transition(clip, transition_duration * 1.2)
                clip = clip.fx(vfx.fadeout, transition_duration)
    
    # Add emotion-specific overlay effects
    add_effect = False
    
    if dominant_emotion == 'joyful':
        # Add sparkles for joyful clips
        print(f"      - Adding sparkle effect (joyful theme)")
        def sparkle_overlay(get_frame, t):
            frame = get_frame(t)
            if t < 2.0:  # Longer duration
                h, w = frame.shape[:2]
                for _ in range(10):  # More sparkles
                    x = random.randint(50, w-50)
                    y = random.randint(50, h-50)
                    frame = create_sparkle(frame, x, y, random.randint(15, 25), 255)
            return frame
        clip = clip.fl(sparkle_overlay)
        add_effect = True
    
    elif dominant_emotion == 'epic' and not is_first and not is_last:
        # Add light leaks for epic clips
        print(f"      - Adding light leak effect (epic theme)")
        def light_leak_overlay(get_frame, t):
            frame = get_frame(t)
            if t < 1.5:
                frame = apply_light_leak_effect(frame.copy(), intensity=0.3)
            return frame
        clip = clip.fl(light_leak_overlay)
        add_effect = True
    
    elif dominant_emotion == 'calm' and clip_index % 2 == 0:
        # Add film grain for calm clips
        print(f"      - Adding film grain effect (calm theme)")
        def grain_overlay(get_frame, t):
            frame = get_frame(t)
            return apply_film_grain(frame.copy(), intensity=0.08)
        clip = clip.fl(grain_overlay)
        add_effect = True
    
    # Add hearts randomly to some clips
    if not is_first and not is_last and clip_index % 4 == 0:
        print(f"      - Adding floating hearts effect")
        def hearts_overlay_mid(get_frame, t):
            frame = get_frame(t)
            if t < 2.0:
                frame = apply_hearts_effect(frame.copy(), t, 2.0)
            return frame
        clip = clip.fl(hearts_overlay_mid)
    
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
            segments = extract_best_segments(video_path, min_duration=2.5, max_segments=1)  # Only 1 segment per video
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
    clip_duration = 2.5
    
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
    
    # 6. Apply diverse transitions and special effects
    step_start = time.time()
    print("\n" + "="*60)
    print("STEP 6: Applying transitions & special effects")
    print("="*60)
    print("  Opening: Flash + rainbow burst + confetti + explosive zoom")
    print("  Finale: Floating hearts + glow + dramatic zoom")
    print("  Transitions: Explosive, slides, zooms, scales")
    print("  Effects: Sparkles, light leaks, film grain, hearts")
    print()
    
    for i in range(len(processed_clips)):
        prev_clip = processed_clips[i-1] if i > 0 else None
        
        processed_clips[i] = apply_diverse_transitions(
            processed_clips[i],
            prev_clip,
            i,
            len(processed_clips),
            dominant_emotion
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
            preset='fast',  # Faster rendering
            bitrate='6000k',  # Higher bitrate for 1080p quality
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