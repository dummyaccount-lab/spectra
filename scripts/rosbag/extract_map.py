import rosbag
import csv
import matplotlib.pyplot as plt
import os
import folium

# Define the path to your rosbag and output CSV file
bag_file_path = 'rosbag_main_vSynced_2024-09-19-14-51-54.bag'


# Initialize an empty list to store the trajectory (latitude, longitude, altitude)
trajectory = []

# Open the bag file
bag = rosbag.Bag(bag_file_path)

# Specify the topic that contains the GPS data
gps_topic = '/ixblue_ins_driver/ix/ins'

for topic, msg, t in bag.read_messages(topics=[gps_topic]):
   
        # Extract latitude and longitude
        latitude = msg.latitude
        longitude = msg.longitude

        # Append the latitude and longitude to the trajectory list
        trajectory.append((latitude, longitude))

# Close the bag file
bag.close()

# Create a folium map centered on the first GPS point
if trajectory:
    initial_lat, initial_lon = trajectory[0]
    map_trajectory = folium.Map(location=[initial_lat, initial_lon], zoom_start=22)

     # Add marker at the start point (green marker)
    folium.Marker([trajectory[0][0], trajectory[0][1]], 
                  popup="Start", 
                  icon=folium.Icon(color="green", icon="play")).add_to(map_trajectory)

    # Add marker at the finish point (red marker)
    folium.Marker([trajectory[-1][0], trajectory[-1][1]], 
                  popup="Finish", 
                  icon=folium.Icon(color="red", icon="stop")).add_to(map_trajectory)

    # Draw a continuous polyline for the trajectory (blue line)
    folium.PolyLine(trajectory, color="blue", weight=2.5, opacity=1).add_to(map_trajectory)

    base_name = os.path.splitext(os.path.basename(bag_file_path))[0]
    date_time = base_name.split('_')[3:]  # Extracting the date and time part
    date_time_str = '-'.join(date_time)  # Joining to form '2024-09-19-14-53-10'
    
    # Construct the HTML file name
    html_file_path = f"trajectory_map_{date_time_str}.html"
    map_trajectory.save(html_file_path)

   

    print(f"Trajectory map saved as '{html_file_path}'. Open this file in a web browser to view the map.")
else:
    print("No GPS data available to plot.")


