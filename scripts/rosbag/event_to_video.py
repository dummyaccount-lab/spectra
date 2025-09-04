import rosbag
import numpy as np
import argparse
import os
from tqdm import tqdm
import cv2

# Function to process accumulated events and write them to an image frame
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


def main(bag_file, topic, output_file):
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with rosbag.Bag(bag_file, 'r') as bag:
        # Get the total number of messages for the specific topic
        total_messages = bag.get_message_count(topic)
        print("Total messages for the topic '{}': {}".format(topic, total_messages))

        # Read the first message to get the dimensions for the video writer
        first_msg = next(bag.read_messages(topics=[topic]))[1]
        width = first_msg.width
        height = first_msg.height

        # Define the codec and create a VideoWriter object for AVI format
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi
        fps = 40  # Define your desired frames per second
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        # Initialize variables to accumulate events and track time
        accumulated_events = []
        frame_duration = 1.0 / fps  # Duration for each frame (1/40th of a second)
        last_frame_time = None  # To track the last frame's timestamp

        # Use tqdm to create a progress bar for the specific topic
        for i, (topic_name, msg, t) in tqdm(enumerate(bag.read_messages(topics=[topic])), total=total_messages, desc="Processing messages"):
            # Assuming msg is of type EventArray
            events = msg.events
            timestamp = t.to_sec()  # Convert timestamp to seconds

            if last_frame_time is None:
                last_frame_time = timestamp  # Initialize for the first message

            # Accumulate events
            accumulated_events.extend(events)

            # Check if enough time has passed for the next frame (1/40th of a second)
            if (timestamp - last_frame_time) >= frame_duration:
                # Process the accumulated events and generate a frame
                frame = process_accumulated_events(accumulated_events, width, height)

                # Write the frame to the video file
                video_writer.write(frame)

                # Reset the event list for the next frame
                accumulated_events = []

                # Update the timestamp for the last frame
                last_frame_time = timestamp

        # After the loop, release the video writer
        video_writer.release()

        print("Video saved to {}".format(output_file))

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert ROS bag events to a video.")
    parser.add_argument('input_bag', type=str, help="Path to the input ROS bag file.")
    parser.add_argument('topic', type=str, help="The topic to read events from.")
    parser.add_argument('output_file', type=str, help="Path to save the output video file.")
    
    args = parser.parse_args()
    bag_file = args.input_bag
    topic = args.topic
    output_file = args.output_file
    
    main(bag_file, topic, output_file)

