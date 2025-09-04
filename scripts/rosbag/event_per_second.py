import h5py
import hdf5plugin
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description='Process an HDF5 file and save a plot.')
parser.add_argument('hdf5_file', type=str, help='Path to the HDF5 file to process')

args = parser.parse_args()

# Open the HDF5 file
hdf5_file = args.hdf5_file

with h5py.File(hdf5_file, 'r') as f:
    # Access the timestamp dataset
    timestamps = f['events/t'][:]  # Get all timestamps

    # Convert timestamps to seconds (assuming timestamps are in microseconds)
    seconds = timestamps // 1_000_000  # Convert to seconds

    # Count the number of events per second
    max_time = seconds.max()  # Find the maximum second
    event_counts = np.zeros(max_time + 1, dtype=int)  # Create an array to hold counts

    for sec in seconds:
        event_counts[sec] += 1  # Increment the count for the corresponding second

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(range(len(event_counts)), event_counts, color='blue', alpha=0.7)
plt.title('Number of Events per Second')
plt.xlabel('Time (seconds)')
plt.ylabel('Number of Events')
plt.xticks(range(len(event_counts)))  # Set x-ticks to correspond to each second
plt.grid(axis='y')

# Save the plot with the same name as the HDF5 file
base_name = os.path.splitext(os.path.basename(hdf5_file))[0]  # Get the base name of the file without extension
output_file = f"{base_name}.png"  # Change extension to .png for the output file
plt.savefig(output_file)  # Save the figure

# Show the plot
plt.show()

# Close the plot
plt.close()

print(f"Plot saved as {output_file}")

