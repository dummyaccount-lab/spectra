#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import threading
import os

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import message_filters
import open3d as o3d

# Parameters
IMAGE_DIR = rospy.get_param('~image_dir', 'save_dir/images')
LIDAR_DIR = rospy.get_param('~lidar_dir', 'save_dir/lidar')
QUEUE_SIZE = rospy.get_param('~queue_size', 1000)
SLOP = rospy.get_param('~slop', 0.05)  # Max time difference allowed (s)
MIN_LIDAR_POINTS = rospy.get_param('~min_lidar_points', 1000)

# Create save directories
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LIDAR_DIR, exist_ok=True)

class DataExtractor:
    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.counter = 0

    def save_pcd(self, points_xyz_i, filename):
        points_xyz_i = np.array(points_xyz_i)
        points = points_xyz_i[:, :3]
        intensities = points_xyz_i[:, 3]

        # Normalize intensity to [0,1] for grayscale
        intensity_normalized = (intensities - intensities.min()) / (intensities.ptp() + 1e-8)
        colors = np.stack([intensity_normalized] * 3, axis=1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.io.write_point_cloud(filename, pcd)
        rospy.loginfo(f"Saved LiDAR (with intensity): {filename}")

    def callback(self, img_msg, lidar_msg):
        try:
            # Convert image
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")

            # Convert point cloud (x, y, z, intensity)
            points = list(pc2.read_points(lidar_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True))
            if len(points) < MIN_LIDAR_POINTS:
                rospy.logwarn("Skipping: Not enough LiDAR points.")
                return

            with self.lock:
                image_path = os.path.join(IMAGE_DIR, f"{self.counter:06d}.png")
                lidar_path = os.path.join(LIDAR_DIR, f"{self.counter:06d}.pcd")

                cv2.imwrite(image_path, cv_image)
                self.save_pcd(points, lidar_path)

                rospy.loginfo(f"[{self.counter:06d}] Saved synchronized image and LiDAR")
                self.counter += 1

        except Exception as e:
            rospy.logerr(f"Callback error: {e}")

def main():
    rospy.init_node('image_lidar_saver_sync', anonymous=True)
    extractor = DataExtractor()

    image_topic = rospy.get_param('~image_topic', '/stereo/left/image_color')
    lidar_topic = rospy.get_param('~lidar_topic', '/ouster/points')

    image_sub = message_filters.Subscriber(image_topic, Image)
    lidar_sub = message_filters.Subscriber(lidar_topic, PointCloud2)

    sync = message_filters.ApproximateTimeSynchronizer(
        [image_sub, lidar_sub], queue_size=QUEUE_SIZE, slop=SLOP, allow_headerless=False)
    sync.registerCallback(extractor.callback)

    rospy.loginfo(f"Synchronized image & LiDAR capture running...\nImages: {IMAGE_DIR}\nLiDAR: {LIDAR_DIR}")
    rospy.spin()

if __name__ == '__main__':
    main()

