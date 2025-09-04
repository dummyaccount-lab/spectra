import rosbag
import os
from tqdm import tqdm
import cv2
import numpy as np
from cv_bridge import CvBridge

def main(bag_file):
    # Extract the base name of the bag file and set the output file path
    bag_filename = os.path.splitext(os.path.basename(bag_file))[0]
    output_file = os.path.join(os.path.dirname(bag_file), f"{bag_filename}.stereo_rgb.avi")

    # Fixed topic names
    left_topic = "/stereo/left/image_color"
    right_topic = "/stereo/right/image_color"

    # Initialize CvBridge
    bridge = CvBridge()

    with rosbag.Bag(bag_file, 'r') as bag:
        # Get the total number of messages in both topics
        total_messages = bag.get_message_count(topic_filters=[left_topic, right_topic])
        print("Total messages in the topics: {}".format(total_messages))

        # Read the first message from each topic to get the dimensions
        first_msg_left = next(bag.read_messages(topics=[left_topic]))[1]
        height = first_msg_left.height
        width = first_msg_left.width

        # Define the codec and create a VideoWriter object for AVI format
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi
        fps = 20  # Set fps to 20
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width * 2, height))

        # Initialize iterators for each topic
        left_messages = bag.read_messages(topics=[left_topic])
        right_messages = bag.read_messages(topics=[right_topic])

        # Use tqdm to create a progress bar
        for _ in tqdm(range(total_messages // 2), desc="Processing frames"):
            # Read the next message from each topic
            _, msg_left, _ = next(left_messages)
            _, msg_right, _ = next(right_messages)

            # Convert ROS Image messages to OpenCV format
            left_frame = bridge.imgmsg_to_cv2(msg_left, desired_encoding="bgr8")
            right_frame = bridge.imgmsg_to_cv2(msg_right, desired_encoding="bgr8")

            # Concatenate the frames horizontally (left on the left, right on the right)
            combined_frame = np.hstack((left_frame, right_frame))

            # Write the combined frame to the video file
            video_writer.write(combined_frame)

        # After the loop, release the video writer
        video_writer.release()

        print(f"Combined video saved to {output_file}")

if __name__ == "__main__":
    # Set up argument parser
    import argparse
    parser = argparse.ArgumentParser(description="Convert ROS bag images from two topics to a combined video.")
    parser.add_argument('input_bag', type=str, help="Path to the input ROS bag file.")
    
    args = parser.parse_args()
    main(args.input_bag)

