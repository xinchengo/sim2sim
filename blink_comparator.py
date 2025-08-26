import cv2
import numpy as np

# Function to create the blink comparator video
def create_blink_comparator_video(image1, image2, output_video_path, frame_rate=30, duration=5, width=640, height=480):
    # Resize images to ensure they fit the desired video dimensions
    image1_resized = cv2.resize(image1, (width, height))
    image2_resized = cv2.resize(image2, (width, height))
    
    # Define the codec and create VideoWriter object for MP4 output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Calculate the number of frames for each image (for alternating effect)
    total_frames = frame_rate * duration
    blink_interval = total_frames // 2

    # Generate frames for the video
    for i in range(total_frames):
        if i % 30 < 15:
            out.write(image1_resized)
        else:
            out.write(image2_resized)
    
    # Release the video writer and close
    out.release()
    print(f"Video saved as {output_video_path}")

# Load the two images
image1 = cv2.imread('mujoco.png')  # Replace with the path to your first image
image2 = cv2.imread('maniskill.png')  # Replace with the path to your second image

# Check if images are loaded correctly
if image1 is None or image2 is None:
    print("Error: Could not load images!")
else:
    # Create the blink comparator video
    create_blink_comparator_video(image1, image2, 'blink_comparator_video.mp4', frame_rate=30, duration=10)
