#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pcl2
import numpy as np
import struct
import open3d as o3d
import ros_numpy

def bin_to_pcd(binFileName):
    """
    Read a binary file and convert it to an Open3D PointCloud.
    Assumes each point is stored as 4 float32 values (x, y, z, intensity).
    """
    size_float = 4
    list_pcd = []
    with open(binFileName, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    return pcd

def callback(data):
    rospy.loginfo('Received a PointCloud2 message')
    
    # Convert ROS PointCloud2 message to a numpy structured array.
    pc = ros_numpy.numpify(data)
    rospy.loginfo(f"PointCloud shape: {pc.shape}")
    
    # Depending on whether the point cloud is organized (2D) or unorganized (1D),
    # we construct a 2D array where each row is [x, y, z, intensity].
    if pc.ndim == 1:
        # Unorganized: pc.shape = (N,)
        points = np.zeros((pc.shape[0], 4), dtype=np.float32)
        points[:, 0] = pc['x']
        points[:, 1] = pc['y']
        points[:, 2] = pc['z']
        points[:, 3] = pc['intensity']
    elif pc.ndim == 2:
        # Organized: pc.shape = (H, W)
        # Flatten the arrays so that we have one row per point.
        num_points = pc.shape[0] * pc.shape[1]
        points = np.zeros((num_points, 4), dtype=np.float32)
        points[:, 0] = pc['x'].flatten()
        points[:, 1] = pc['y'].flatten()
        points[:, 2] = pc['z'].flatten()
        points[:, 3] = pc['intensity'].flatten()
    else:
        rospy.logerr("Unexpected point cloud shape.")
        return

    global pc_number
    pc_number += 1

    # Set your file paths here
    bin_file = f'./bin/{pc_number}.bin'       # Folder for .bin files
    pcd_file = f'./pcd/{pc_number}.pcd'    # Folder for .pcd files

    # Save the binary file
    points.tofile(bin_file)
    rospy.loginfo(f"Saved {points.shape[0]} points to {bin_file}")

    # Convert the .bin file to a point cloud (only x, y, z used)
    pcd = bin_to_pcd(bin_file)
    o3d.io.write_point_cloud(pcd_file, pcd)
    rospy.loginfo(f"Saved point cloud to {pcd_file}")

def listener():
    global pc_number
    pc_number = 0  # Initialize frame counter
    
    rospy.init_node('pointcloud_to_bin_file', anonymous=True)
    
    # Subscribe to the input PointCloud2 topic (adjust topic as needed)
    input_cloud = '/ouster/points'
    rospy.Subscriber(input_cloud, PointCloud2, callback)
    
    # (Optional) Create a publisher for an output topic if you wish to republish the cloud.
    # pub = rospy.Publisher('/output_cloud', PointCloud2, queue_size=10)
    
    rospy.spin()

if __name__ == '__main__':
    listener()
