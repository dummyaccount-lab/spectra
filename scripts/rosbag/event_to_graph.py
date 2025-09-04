import rosbag
import argparse
import os
import pandas as pd
from tqdm import tqdm

def main(bag_file, topic, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    with rosbag.Bag(bag_file, 'r') as bag:
        # Get the total number of messages for the specific topic
        total_messages = bag.get_message_count(topic)
        print("Total messages for the topic '{}': {}".format(topic, total_messages))

        # Initialize variables to track event rates and timestamps
        current_second = None
        event_count = 0
        event_rates = []
        timestamps = []

        # Use tqdm to create a progress bar for the specific topic
        for i, (topic_name, msg, t) in tqdm(enumerate(bag.read_messages(topics=[topic])), total=total_messages, desc="Processing messages"):
            # Assuming msg is of type EventArray
            events = msg.events
            timestamp = t.to_sec()  # Convert timestamp to seconds

            # Determine the current second
            second = int(timestamp)

            # Check if we are still in the same second
            if current_second is None:
                current_second = second

            if second == current_second:
                # Accumulate events for the current second
                event_count += len(events)
            else:
                # Store the event rate for the previous second
                event_rates.append(event_count)
                timestamps.append(current_second)

                # Reset for the new second
                current_second = second
                event_count = len(events)

        # After processing all messages, ensure the last second's data is included
        if event_count > 0:
            event_rates.append(event_count)
            timestamps.append(current_second)

        # Save the results to an Excel file
        data = {'Timestamp (s)': timestamps, 'Event Rate (events/s)': event_rates}
        df = pd.DataFrame(data)
        output_file = os.path.join(output_dir, 'event_rate_data.xlsx')
        df.to_excel(output_file, index=False)
        print("Event rate data saved to {}".format(output_file))

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Extract event rate per second from ROS bag and save to Excel.")
    parser.add_argument('input_bag', type=str, help="Path to the input ROS bag file.")
    parser.add_argument('topic', type=str, help="The topic to read events from.")
    parser.add_argument('output_dir', type=str, help="Directory to save the Excel file.")
    
    args = parser.parse_args()
    bag_file = args.input_bag
    topic = args.topic
    output_dir = args.output_dir
    
    main(bag_file, topic, output_dir)

