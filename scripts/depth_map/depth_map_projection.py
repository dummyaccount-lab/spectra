import numpy as np
import open3d as o3d
import cv2
import os
from scipy.spatial.transform import Rotation as R

def read_imu_file(file_path):
    imu_data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            values = line.strip().split()
            if len(values) == 8:
                imu_data.append([float(v) for v in values])
    return np.array(imu_data)

T_imu_to_camera_init = np.array([
    [-0.02561535, -0.99958868, -0.01287125,  0.20361100],
    [-0.06779258,  0.01458285, -0.99759266, -0.05446648],
    [ 0.99737086, -0.02468116, -0.06813827,  0.20798620],
    [ 0.00000000,  0.00000000,  0.00000000,  1.00000000]
])

rotation_offset = [0.0, 0.0, 0.0]
translation_offset = [0.0, 0.0, 0.0]

button_width = 30
button_height = 30
gap = 20
start_x = 20
start_y = 20

param_names = ['Rx', 'Ry', 'Rz', 'Tx', 'Ty', 'Tz']
rects = {}

def update_matrix():
    rx, ry, rz = np.deg2rad(rotation_offset)
    tx, ty, tz = translation_offset
    R_total = R.from_euler('xyz', [rx, ry, rz]).as_matrix()
    T_updated = T_imu_to_camera_init.copy()
    T_updated[:3, :3] = R_total @ T_updated[:3, :3]
    T_updated[:3, 3] += [tx, ty, tz]
    return T_updated

def print_matrix_to_terminal(T):
    print("Updated transformation matrix:")
    for row in T:
        print(" ".join(f"{v: .8f}" for v in row))
    print("\n" + "-"*50 + "\n")

def draw_controls_window():
    w = 2 * button_width + 120
    h = len(param_names) * (button_height + gap) + gap
    img = np.ones((h, w, 3), dtype=np.uint8) * 220

    for i, name in enumerate(param_names):
        y = gap + i * (button_height + gap)

        minus_rect = (start_x, y, start_x + button_width, y + button_height)
        plus_rect = (start_x + button_width + 80, y, start_x + 2 * button_width + 80, y + button_height)
        rects[f"{name}_minus"] = minus_rect
        rects[f"{name}_plus"] = plus_rect

        cv2.rectangle(img, minus_rect[:2], minus_rect[2:], (180, 180, 180), -1)
        cv2.putText(img, "-", (start_x + 10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.rectangle(img, plus_rect[:2], plus_rect[2:], (180, 180, 180), -1)
        cv2.putText(img, "+", (plus_rect[0] + 7, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        val = rotation_offset[i] if i < 3 else translation_offset[i - 3]
        cv2.putText(img, f"{name}: {val:.3f}", (start_x + button_width + 10, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow("Controls", img)

def on_mouse(event, x, y, flags, param):
    global rotation_offset, translation_offset, paused_frame_data
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, name in enumerate(param_names):
            if rects[f"{name}_minus"][0] <= x <= rects[f"{name}_minus"][2] and rects[f"{name}_minus"][1] <= y <= rects[f"{name}_minus"][3]:
                if i < 3:
                    rotation_offset[i] -= 0.1
                else:
                    translation_offset[i - 3] -= 0.01
                draw_controls_window()
                print_matrix_to_terminal(update_matrix())
                if paused and paused_frame_data:
                    redraw_paused_frame(paused_frame_data)
                return

            if rects[f"{name}_plus"][0] <= x <= rects[f"{name}_plus"][2] and rects[f"{name}_plus"][1] <= y <= rects[f"{name}_plus"][3]:
                if i < 3:
                    rotation_offset[i] += 0.1
                else:
                    translation_offset[i - 3] += 0.01
                draw_controls_window()
                print_matrix_to_terminal(update_matrix())
                if paused and paused_frame_data:
                    redraw_paused_frame(paused_frame_data)
                return

def find_closest_image(timestamp, image_list):
    timestamp_ns = int(timestamp * 1e9)
    return min(image_list, key=lambda x: abs(int(x[0]) - timestamp_ns))

def redraw_paused_frame(data):
    timestamp = data["timestamp"]
    vehicle_pos = data["vehicle_pos"]
    nearby_points = data["nearby_points"]
    filename = data["filename"]

    T_imu_to_camera = update_matrix()
    imu_entry = data["imu_entry"]

    _, x, y, z, q_x, q_y, q_z, q_w = imu_entry
    R_imu = R.from_quat([q_x, q_y, q_z, q_w]).as_matrix()
    T_imu_to_world = np.eye(4)
    T_imu_to_world[:3, :3] = R_imu
    T_imu_to_world[:3, 3] = [x, y, z]
    T_world_to_camera = T_imu_to_camera @ np.linalg.inv(T_imu_to_world)

    points_hom = np.hstack((nearby_points, np.ones((nearby_points.shape[0], 1))))
    points_cam = (T_world_to_camera @ points_hom.T).T
    mask_z = points_cam[:, 2] > 0
    points_valid = points_cam[mask_z, :3]

    if points_valid.shape[0] == 0:
        return

    x_proj = points_valid[:, 0] / points_valid[:, 2]
    y_proj = points_valid[:, 1] / points_valid[:, 2]
    u = (K[0, 0] * x_proj + K[0, 2]).astype(int)
    v = (K[1, 1] * y_proj + K[1, 2]).astype(int)

    mask_uv = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
    u_valid = u[mask_uv]
    v_valid = v[mask_uv]
    z_valid = points_valid[mask_uv, 2]
    if z_valid.size == 0:
        return

    img = cv2.imread(os.path.join(rgb_image_dir, filename))
    if img is None:
        return

    sorted_indices = np.argsort(z_valid)
    u_sorted = u_valid[sorted_indices]
    v_sorted = v_valid[sorted_indices]
    z_sorted = z_valid[sorted_indices]

    z_min, z_max = z_sorted.min(), z_sorted.max()
    depth_colors = (255 * (1 - (z_sorted - z_min) / (z_max - z_min + 1e-5))).astype(np.uint8)

    overlay = img.copy()
    for i in range(len(u_sorted)):
        color = (int(depth_colors[i]), 0, 255 - int(depth_colors[i]))
        cv2.circle(overlay, (u_sorted[i], v_sorted[i]), 2, color, -1)

    blended = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    cv2.imshow("Live Stream", blended)

def main():
    global paused_frame_data, paused, K, image_width, image_height, rgb_image_dir, imu_data
    imu_file_path = "traj.txt"
    pcd_file_path = "map-76.pcd"
    rgb_image_dir = "./images/"

    imu_data = read_imu_file(imu_file_path)
    if imu_data.shape[1] != 8:
        raise ValueError("IMU file must have 8 columns.")

    point_cloud = o3d.io.read_point_cloud(pcd_file_path)
    if point_cloud.is_empty():
        raise ValueError("PCD file is empty.")
    points_all = np.asarray(point_cloud.points)

    K = np.array([
        [1685.65008, 0, 629.89747],
        [0, 1686.0903, 373.90386],
        [0, 0, 1]
    ])
    image_width, image_height = 1280, 720

    rgb_images = []
    for filename in os.listdir(rgb_image_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            try:
                ts = float(os.path.splitext(filename)[0])
                rgb_images.append((ts, filename))
            except ValueError:
                continue
    rgb_images.sort()

    draw_controls_window()
    cv2.setMouseCallback("Controls", on_mouse)

    paused = False
    paused_frame_data = None
    imu_index = 0
    n_imu = imu_data.shape[0]

    while True:
        if not paused:
            imu_entry = imu_data[imu_index]
            imu_index += 1
            if imu_index >= n_imu:
                break

            timestamp, x, y, z, q_x, q_y, q_z, q_w = imu_entry
            T_imu_to_camera = update_matrix()

            R_imu = R.from_quat([q_x, q_y, q_z, q_w]).as_matrix()
            T_imu_to_world = np.eye(4)
            T_imu_to_world[:3, :3] = R_imu
            T_imu_to_world[:3, 3] = [x, y, z]

            T_world_to_camera = T_imu_to_camera @ np.linalg.inv(T_imu_to_world)

            vehicle_pos = T_imu_to_world[:3, 3]
            distances = np.linalg.norm(points_all - vehicle_pos, axis=1)
            nearby_points = points_all[distances < 75.0]

            if nearby_points.shape[0] == 0:
                continue

            points_hom = np.hstack((nearby_points, np.ones((nearby_points.shape[0], 1))))
            points_cam = (T_world_to_camera @ points_hom.T).T
            mask_z = points_cam[:, 2] > 0
            points_valid = points_cam[mask_z, :3]
            if points_valid.shape[0] == 0:
                continue

            x_proj = points_valid[:, 0] / points_valid[:, 2]
            y_proj = points_valid[:, 1] / points_valid[:, 2]
            u = (K[0, 0] * x_proj + K[0, 2]).astype(int)
            v = (K[1, 1] * y_proj + K[1, 2]).astype(int)

            mask_uv = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
            u_valid = u[mask_uv]
            v_valid = v[mask_uv]
            z_valid = points_valid[mask_uv, 2]
            if z_valid.size == 0:
                continue

            closest_ts, filename = find_closest_image(timestamp, rgb_images)
            img = cv2.imread(os.path.join(rgb_image_dir, filename))
            if img is None:
                continue

            sorted_indices = np.argsort(z_valid)
            u_sorted = u_valid[sorted_indices]
            v_sorted = v_valid[sorted_indices]
            z_sorted = z_valid[sorted_indices]

            z_min, z_max = z_sorted.min(), z_sorted.max()
            depth_colors = (255 * (1 - (z_sorted - z_min) / (z_max - z_min + 1e-5))).astype(np.uint8)

            overlay = img.copy()
            for i in range(len(u_sorted)):
                color = (int(depth_colors[i]), 0, 255 - int(depth_colors[i]))
                cv2.circle(overlay, (u_sorted[i], v_sorted[i]), 2, color, -1)

            blended = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
            cv2.imshow("Live Stream", blended)

            paused_frame_data = {
                "timestamp": timestamp,
                "vehicle_pos": vehicle_pos,
                "nearby_points": nearby_points,
                "filename": filename,
                "imu_entry": imu_entry
            }

        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break
        elif key == ord(' '):
            paused = not paused

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
