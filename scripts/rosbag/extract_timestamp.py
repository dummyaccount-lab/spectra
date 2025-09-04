import rosbag
import os
import sys

def process_bag_file(bag_file_path):
    print(f"Processing {bag_file_path}...")

    # Initialize an empty list to store the timestamps (ts)
    timestamps = []

    # Open the bag file
    bag = rosbag.Bag(bag_file_path)

    # Specify the topic that contains the trigger_event data
    event_topic = '/prophesee/camera1_master/trigger_event'

    for topic, msg, t in bag.read_messages(topics=[event_topic]):
        # Extract the ts value (in microseconds)
        ts_value = msg.ts  # Assuming it's in microseconds

        # Store the ts value
        timestamps.append(ts_value)

    # Close the bag file
    bag.close()

    # Save the timestamps to a .txt file
    if timestamps:
        # Extract the date and time from the bag file name
        base_name = os.path.splitext(os.path.basename(bag_file_path))[0]
        
        # Construct the .txt file name
        txt_file_path = f"trigger_event_timestamps_{base_name}.txt"
        
        with open(txt_file_path, 'w') as txt_file:
            # Write each ts value to the file without a header
            for ts_value in timestamps:
                txt_file.write(f"{ts_value}\n")

        print(f"Timestamps saved as '{txt_file_path}'.")
    else:
        print("No event data available to save.")

def main():
    # Check if a path argument is provided
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_bag_or_folder>")
        sys.exit(1)

    input_path = sys.argv[1]

    # Check if the input path is a directory or a file
    if os.path.isdir(input_path):
        # Loop through all .bag files in the directory
        for file_name in os.listdir(input_path):
            if file_name.endswith('.bag'):
                process_bag_file(os.path.join(input_path, file_name))
    elif os.path.isfile(input_path) and input_path.endswith('.bag'):
        # Process the single .bag file
        process_bag_file(input_path)
    else:
        print("Invalid input. Please provide a valid .bag file or directory containing .bag files.")

if __name__ == "__main__":
    main()

