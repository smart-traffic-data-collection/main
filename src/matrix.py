import os
import glob
import pandas as pd
import numpy as np

# Define path to the newly created tracking results
tracking_results_dir = "/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/models/finals/tracking_eval"

# Initialize list to store our raw parsed data
jitter_data = []

# 1. Recursively find all generated tracking .txt files
print(f"Scanning for tracking label files in:\n{tracking_results_dir}...\n")
label_files = glob.glob(os.path.join(tracking_results_dir, "**", "labels", "*.txt"), recursive=True)

if not label_files:
    print("❌ Error: No tracking files found! Check the directory path.")
    exit()

print(f"✅ Found {len(label_files)} frames containing tracking data.\n")
print("Parsing trajectories. This might take a few seconds...")

# 2. Parse the files
for filepath in label_files:
    # Extract the sequence/camera name from the directory path
    parts = filepath.split(os.sep)
    sequence_folder = parts[-3] 
    
    # Read the text file
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    # Standard YOLO track format: [class] [x] [y] [w] [h] [conf] [track_id]
    for line in lines:
        data = line.strip().split()
        if len(data) >= 7:
            cls_id = int(data[0])
            w = float(data[3])
            h = float(data[4])
            track_id = int(data[6])
            
            # Store data for calculating Jitter
            jitter_data.append({
                'Sequence': sequence_folder,
                'Track_ID': track_id,
                'Width': w,
                'Height': h
            })

# 3. Calculate Smoothness / Jitter Metrics
df = pd.DataFrame(jitter_data)

# We only want to analyze tracks that exist for a reasonable amount of time.
# Filter out vehicles that only appeared for 1-5 frames (too short to measure jitter).
track_counts = df.groupby(['Sequence', 'Track_ID']).size()
valid_tracks = track_counts[track_counts > 5].reset_index()

filtered_df = pd.merge(df, valid_tracks[['Sequence', 'Track_ID']], on=['Sequence', 'Track_ID'])

# Calculate the standard deviation (jitter) for each vehicle
jitter_stats = filtered_df.groupby(['Sequence', 'Track_ID']).agg(
    W_Std=('Width', 'std'),
    H_Std=('Height', 'std'),
    Frame_Count=('Width', 'count')
).reset_index()

# Overall Average Jitter across the entire validation set
avg_width_jitter = jitter_stats['W_Std'].mean()
avg_height_jitter = jitter_stats['H_Std'].mean()

# Output the final Matrix Metrics
print("="*50)
print("🎯 THERMAL TRACKING SMOOTHNESS METRICS 🎯")
print("="*50)
print(f"Total Valid Vehicle Tracks Analyzed: {len(jitter_stats)}")
print(f"Average Bounding Box Width Jitter:  {avg_width_jitter:.5f}")
print(f"Average Bounding Box Height Jitter: {avg_height_jitter:.5f}")
print("="*50)

# Grading Logic (Specifically tuned for normalized coordinates 0.0 - 1.0)
if avg_width_jitter < 0.015 and avg_height_jitter < 0.015:
    print("\n✅ Verdict: EXCELLENT. \nBounding boxes are highly stable. The model handles thermal crossover beautifully. Ready for RGB Sensor Fusion.")
elif avg_width_jitter < 0.035:
    print("\n⚠️ Verdict: ACCEPTABLE. \nMinor thermal shifting present. The ByteTrack Kalman filter is doing most of the heavy lifting.")
else:
    print("\n❌ Verdict: POOR. \nSevere bounding box jitter. Model")