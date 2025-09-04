import os
import numpy as np
import open3d as o3d

# Folder containing .bin files
bin_folder = "./120113/velodyne"
# Output folder for .pcd files
pcd_folder = "./pcd_scans/120113"

os.makedirs(pcd_folder, exist_ok=True)

# Get list of already converted .pcd files (without extension)
existing_pcds = {f.replace(".pcd", "") for f in os.listdir(pcd_folder) if f.endswith(".pcd")}

for filename in sorted(os.listdir(bin_folder)):
    if filename.endswith(".bin"):
        base_name = filename.replace(".bin", "")
        if base_name in existing_pcds:
            print(f"Skipping {filename} (already converted)")
            continue

        bin_path = os.path.join(bin_folder, filename)
        pcd_name = base_name + ".pcd"
        pcd_path = os.path.join(pcd_folder, pcd_name)

        # Load point cloud from bin file
        try:
            points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 6)
        except ValueError as e:
            print(f"Error reading {filename}: {e}")
            continue

        # Convert to Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # x, y, z

        # Use intensity as grayscale color
        intensities = points[:, 3]
        intensities = (intensities - intensities.min()) / (intensities.ptp() + 1e-6)
        colors = np.tile(intensities.reshape(-1, 1), (1, 3))
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Save to PCD
        o3d.io.write_point_cloud(pcd_path, pcd)
        print(f"Converted {filename} -> {pcd_name}")

print("Finished converting remaining .bin files.")

