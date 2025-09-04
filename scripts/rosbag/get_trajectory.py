import rosbag
import os
import sys

def process_bag_file(bag_file_path):
    print(f"Processing {bag_file_path}...")

    # Initialize an empty list to store the trajectory (timestamp, latitude, longitude)
    trajectory = []

    # Open the bag file
    bag = rosbag.Bag(bag_file_path)

    # Specify the topic that contains the GPS data
    gps_topic = '/ixblue_ins_driver/ix/ins'

    for topic, msg, t in bag.read_messages(topics=[gps_topic]):
        # Extract latitude, longitude, and timestamp
        latitude = msg.latitude
        longitude = msg.longitude
        
        # Create the ROS timestamp format (seconds and nanoseconds)
        timestamp_sec = msg.header.stamp.secs
        timestamp_nsec = msg.header.stamp.nsecs
        timestamp_ros = f"{timestamp_sec}.{timestamp_nsec:09d}"
        
        # Append the timestamp, latitude, and longitude to the trajectory list
        trajectory.append((timestamp_ros, latitude, longitude))

    # Close the bag file
    bag.close()

    # Save the trajectory to a .txt file
    if trajectory:
        # Extract the date and time from the bag file name
        base_name = os.path.splitext(os.path.basename(bag_file_path))[0]
        
        # Construct the .txt file name
        txt_file_path = f"trajectory_{base_name}.txt"
        
        with open(txt_file_path, 'w') as txt_file:
            # Write the header
            txt_file.write("Timestamp,Latitude,Longitude\n")
            
            # Write the trajectory data
            for timestamp, lat, lon in trajectory:
                txt_file.write(f"{timestamp},{lat},{lon}\n")

        print(f"Trajectory data saved as '{txt_file_path}'.")
    else:
        print("No GPS data available to save.")

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

