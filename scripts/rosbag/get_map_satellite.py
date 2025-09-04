import rosbag
import os
import sys
import folium

def process_bag_file(bag_file_path):
    print(f"Processing {bag_file_path}...")

    # Initialize an empty list to store the trajectory (latitude, longitude, timestamp)
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
        
        # Append the latitude, longitude, and timestamp to the trajectory list
        trajectory.append((latitude, longitude, timestamp_ros))

    # Close the bag file
    bag.close()

    # Create a folium map centered on the first GPS point
    if trajectory:
        initial_lat, initial_lon, _ = trajectory[0]
        #map_trajectory = folium.Map(location=[initial_lat, initial_lon], zoom_start=22)
        map_trajectory = folium.Map(
	    location=[initial_lat, initial_lon],
	    zoom_start=22,
	    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
	    attr='Esri World Imagery'
	)

        # Add marker at the start point (green marker)
        folium.Marker(
            [initial_lat, initial_lon],
            popup="Start",
            icon=folium.Icon(color="green", icon="play")
        ).add_to(map_trajectory)

        # Add marker at the finish point (red marker)
        final_lat, final_lon, _ = trajectory[-1]
        folium.Marker(
            [final_lat, final_lon],
            popup="Finish",
            icon=folium.Icon(color="red", icon="stop")
        ).add_to(map_trajectory)

        # Draw the trajectory with timestamp popups
        for lat, lon, timestamp in trajectory:
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color="blue",
                fill=True,
                fill_color="blue",
                fill_opacity=0.7,
                popup=f"Timestamp: {timestamp}"  # ROS timestamp format
            ).add_to(map_trajectory)

        # Extract the date and time from the bag file name
        base_name = os.path.splitext(os.path.basename(bag_file_path))[0]
        

        # Construct the HTML file name
        html_file_path = f"trajectory_map_{base_name}.html"
        map_trajectory.save(html_file_path)

        print(f"Trajectory map saved as '{html_file_path}'. Open this file in a web browser to view the map.")
    else:
        print("No GPS data available to plot.")

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

