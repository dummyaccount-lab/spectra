import rosbag
import numpy as np
import argparse
import os
from tqdm import tqdm
import cv2

# Function to process events and write them to an image frame for each topic
def process_accumulated_events(accumulated_events, width, height):
    # Create a blank image with white background
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    # Draw accumulated events on the frame
    for event in accumulated_events:
        if event.polarity:  # Positive event
            frame[event.y, event.x] = [255, 0, 0]  # Blue pixel for positive events (BGR format)
        else:  # Negative event
            frame[event.y, event.x] = [0, 0, 255]  # Red pixel for negative events (BGR format)

    return frame

# Function to combine 2 frames (left and right) into a single frame
def combine_frames(left_frame, right_frame):
    # Concatenate the two frames horizontally to create a 2-panel view
    combined_frame = np.hstack((left_frame, right_frame))
    return combined_frame

def main(bag_file, topics, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    # Extract the base name of the bag file to use for the output video
    bag_filename = os.path.splitext(os.path.basename(bag_file))[0]
    output_video_path = os.path.join(output_dir, f"{bag_filename}.avi")

    with rosbag.Bag(bag_file, 'r') as bag:
        # Get the total number of messages to set up the progress bar
        total_messages = bag.get_message_count()
        print("Total messages in the bag: {}".format(total_messages))

        # Read the first message from the first topic to get the dimensions for the video writer
        first_msg_1 = next(bag.read_messages(topics=[topics[0]]))[1]
        width = first_msg_1.width
        height = first_msg_1.height

        # Define the codec and create a VideoWriter object for AVI format
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi
        fps = 30  # Define your desired frames per second (30 Hz)
        # Set video writer dimensions to be the combined width (two frames) and the original height
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width * 2, height))

        # Initialize variables to accumulate events and track time for each topic
        accumulated_events = {topic: [] for topic in topics}
        frame_duration = 1.0 / fps  # 1/30th of a second
        last_frame_time = None  # To track the last frame's timestamp

        # Use tqdm to create a progress bar
        for i, (topic_name, msg, t) in tqdm(enumerate(bag.read_messages(topics=topics)), total=total_messages, desc="Processing messages"):
            events = msg.events
            timestamp = t.to_sec()  # Convert timestamp to seconds

            if last_frame_time is None:
                last_frame_time = timestamp  # Initialize for the first message

            # Accumulate events for each topic
            accumulated_events[topic_name].extend(events)

            # Check if enough time has passed for the next frame (1/30th of a second)
            if (timestamp - last_frame_time) >= frame_duration:
                # Process the accumulated events for each topic without resizing
                left_frame = process_accumulated_events(accumulated_events[topics[0]], width, height)
                right_frame = process_accumulated_events(accumulated_events[topics[1]], width, height)

                # Combine the two frames into one
                combined_frame = combine_frames(left_frame, right_frame)

                # Write the combined frame to the video file
                video_writer.write(combined_frame)

                # Reset the event list for each topic
                accumulated_events = {topic: [] for topic in topics}

                # Update the timestamp for the last frame
                last_frame_time = timestamp

        # After the loop, release the video writer
        video_writer.release()

        print(f"Video saved to {output_video_path}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert ROS bag events to a 2-panel video.")
    parser.add_argument('input_bag', type=str, help="Path to the input ROS bag file.")
    parser.add_argument('topics', nargs=2, type=str, help="The 2 topics to read events from.")
    parser.add_argument('output_dir', type=str, help="Directory to save the video.")
    
    args = parser.parse_args()
    bag_file = args.input_bag
    topics = args.topics
    output_dir = args.output_dir
    
    main(bag_file, topics, output_dir)

