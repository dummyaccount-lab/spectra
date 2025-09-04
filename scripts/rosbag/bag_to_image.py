import os
import rosbag
import cv2
from cv_bridge import CvBridge
import argparse
from tqdm import tqdm

def save_images_from_bag(bag_file, image_topic, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize CvBridge
    bridge = CvBridge()

    # Open the bag file to read messages
    with rosbag.Bag(bag_file, 'r') as bag:
        total_messages = bag.get_message_count(topic_filters=[image_topic])
        print(f"Total images to process: {total_messages}")

        # Process each message and update the progress bar
        with tqdm(total=total_messages, desc="Saving images", smoothing=0) as pbar:
            for _, msg, t in bag.read_messages(topics=[image_topic]):
                # Convert ROS Image message to OpenCV format
                image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                
                # Generate a filename based on the timestamp
                timestamp = t.to_nsec()  # Use nanoseconds for more precision
                filename = os.path.join(output_dir, f"{timestamp}.png")
                
                # Save the image
                cv2.imwrite(filename, image)
                
                # Update the progress bar
                pbar.update(1)

    print("All images saved successfully.")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag and save them as files named with the timestamp.")
    parser.add_argument('bag_file', type=str, help="Path to the input ROS bag file.")
    parser.add_argument('image_topic', type=str, help="The topic name of the image messages.")
    parser.add_argument('output_dir', type=str, help="Directory to save the extracted images.")
    
    args = parser.parse_args()
    save_images_from_bag(args.bag_file, args.image_topic, args.output_dir)

