import cv2
import argparse

# Function to concatenate two videos side by side
def concatenate_videos(video1_path, video2_path, output_path):
    # Open the first video
    cap1 = cv2.VideoCapture(video1_path)
    # Open the second video
    cap2 = cv2.VideoCapture(video2_path)

    # Get the width and height of the first video
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get the width and height of the second video
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Check if both videos have the same height
    if height1 != height2:
        print("The videos must have the same height.")
        return

    # Define the codec and create VideoWriter object for AVI output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, cap1.get(cv2.CAP_PROP_FPS), (width1 + width2, height1))

    while True:
        # Read a frame from the first video
        ret1, frame1 = cap1.read()
        # Read a frame from the second video
        ret2, frame2 = cap2.read()

        # Break the loop if either video ends
        if not ret1 or not ret2:
            break

        # Concatenate the two frames side by side
        combined_frame = cv2.hconcat([frame1, frame2])

        # Write the combined frame to the output video
        out.write(combined_frame)

    # Release everything if job is finished
    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()
    print("Videos concatenated successfully!")

# Main function to parse arguments and call the concatenate function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate two videos side by side.")
    parser.add_argument("video1", help="Path to the first input video.")
    parser.add_argument("video2", help="Path to the second input video.")
    parser.add_argument("output", help="Path to the output concatenated video (should be .avi).")

    args = parser.parse_args()

    # Ensure the output file has .avi extension
    if not args.output.lower().endswith('.avi'):
        print("Output file must have .avi extension.")
    else:
        concatenate_videos(args.video1, args.video2, args.output)

