import parselmouth
from parselmouth.praat import call
import numpy as np

def extract_voice_features(audio_file_path):
    """
    Takes a .wav file, analyzes the voice using Praat, 
    and returns the 4 required features.
    """
    try:
        sound = parselmouth.Sound(audio_file_path)
        
        pitch = call(sound, "To Pitch", 0.0, 75, 600)
        mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
        
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        
        local_shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        
        if np.isnan(mean_pitch): mean_pitch = 0
        if np.isnan(local_jitter): local_jitter = 0
        if np.isnan(local_shimmer): local_shimmer = 0
        if np.isnan(hnr): hnr = 0
        
        return [mean_pitch, local_jitter * 100, local_shimmer * 100, hnr]
        
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return None