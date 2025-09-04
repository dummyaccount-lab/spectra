import rosbag
import csv
import matplotlib.pyplot as plt
import os

# Define the path to your rosbag and output CSV file
bag_file_path = 'rosbag_main_vSynced_2024-09-19-14-31-00.bag'
base_name = os.path.splitext(os.path.basename(bag_file_path))[0]
date_time = base_name.split('_')[3:]  # Extracting the date and time part
date_time_str = '-'.join(date_time)  # Joining to form '2024-09-19-14-53-10'
    
# Construct the CSV file name
csv_file_path = f"trajectory_map_{date_time_str}.csv"

# Initialize an empty list to store the trajectory (latitude, longitude, altitude)
trajectory = []

# Open the bag file
with rosbag.Bag(bag_file_path, 'r') as bag:
    # Specify the topic that contains the GPS data
    gps_topic = '/ixblue_ins_driver/standard/navsatfix'

    # Read messages from the bag file
    for topic, msg, t in bag.read_messages(topics=[gps_topic]):
        # Extract latitude, longitude, and altitude
        latitude = msg.latitude
        longitude = msg.longitude
        altitude = msg.altitude
        
        # Append the data to the trajectory list
        trajectory.append((latitude, longitude, altitude))

# Write the trajectory data to a CSV file
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Latitude', 'Longitude', 'Altitude'])  # Write header
    csv_writer.writerows(trajectory)  # Write data

print(f"Trajectory saved to {csv_file_path}")

# Plot the trajectory (Latitude vs Longitude)
latitude_values = [point[0] for point in trajectory]
longitude_values = [point[1] for point in trajectory]

plt.figure()
plt.plot(longitude_values, latitude_values, marker='o', linestyle='-', color='b')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Trajectory Plot (Latitude vs Longitude)')
plt.grid(True)
plt.show()