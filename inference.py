from ultralytics import YOLO
import cv2
import numpy as np
import torch
import os

# Add the required global to safe globals
torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])

# Load the trained model
model = YOLO('best.pt')

def process_image(image_path):
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist")
        return
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Run YOLOv8 inference on the image
    results = model(image)
    
    # Visualize the results on the image
    annotated_image = results[0].plot()
    
    # Save the annotated image
    output_path = 'output_' + os.path.basename(image_path)
    cv2.imwrite(output_path, annotated_image)
    
    # Display the image
    cv2.imshow("YOLOv8 Inference", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"Image processing completed. Output saved to {output_path}")

def process_video(video_path):
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"Error: File {video_path} does not exist")
        return
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer object
    output_path = 'output_' + os.path.basename(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    while cap.isOpened():
        success, frame = cap.read()
        
        if not success:
            break
        
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Write the frame to the output video
        out.write(annotated_frame)
        
        # Display the frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Video processing completed. Output saved to {output_path}")

# Main execution
if __name__ == "__main__":
    # List of supported image formats
    image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    # List of supported video formats
    video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    # Get the input file path
    input_path = input("Enter the path to your image or video file: ")
    
    # Get file extension
    file_ext = os.path.splitext(input_path)[1].lower()
    
    # Process based on file type
    if file_ext in image_formats:
        process_image(input_path)
    elif file_ext in video_formats:
        process_video(input_path)
    else:
        print(f"Error: Unsupported file format. Supported formats are:")
        print("Images:", ", ".join(image_formats))
        print("Videos:", ", ".join(video_formats))
