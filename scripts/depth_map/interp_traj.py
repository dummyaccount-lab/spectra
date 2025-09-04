import os
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.signal import savgol_filter

def load_trajectory(file_path):
    times, positions, quats = [], [], []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            vals = line.strip().split()
            times.append(float(vals[0]))
            positions.append([float(x) for x in vals[1:4]])
            quats.append([float(x) for x in vals[4:8]])
    return np.array(times), np.array(positions), np.array(quats)

def smooth_positions(positions, window_length=7, polyorder=2):
    if positions.shape[0] < window_length:
        return positions
    return savgol_filter(positions, window_length, polyorder, axis=0)

def slerp_quaternion(q0, q1, alpha):
    r = R.from_quat([q0, q1])
    return Slerp([0, 1], r)([alpha])[0].as_quat()

def interpolate_pose(t, times, positions, quats):
    if t <= times[0]:
        dt = times[1] - times[0]
        velocity = (positions[1] - positions[0]) / dt
        extrap_pos = positions[0] + velocity * (t - times[0])
        return extrap_pos, quats[0]
    elif t >= times[-1]:
        dt = times[-1] - times[-2]
        velocity = (positions[-1] - positions[-2]) / dt
        extrap_pos = positions[-1] + velocity * (t - times[-1])
        return extrap_pos, quats[-1]

    i = np.searchsorted(times, t)
    t0, t1 = times[i - 1], times[i]
    alpha = (t - t0) / (t1 - t0)
    p0, p1 = positions[i - 1], positions[i]
    q0, q1 = quats[i - 1], quats[i]

    interp_pos = (1 - alpha) * p0 + alpha * p1
    interp_quat = slerp_quaternion(q0, q1, alpha)
    return interp_pos, interp_quat

def extract_timestamp(file_name):
    try:
        base = os.path.splitext(file_name)[0]
        return float(base) / 1e9 if len(base) > 10 else float(base)
    except Exception as e:
        raise ValueError(f"Invalid filename timestamp: {file_name}")

def synchronize_poses(traj_file, image_dir, output_file):
    traj_times, traj_pos, traj_quat = load_trajectory(traj_file)
    traj_pos = smooth_positions(traj_pos)

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    output_lines = ["#timestamp x y z q_x q_y q_z q_w"]
    skipped = 0

    for fname in image_files:
        try:
            t_img = extract_timestamp(fname)
        except ValueError:
            continue

        pos, quat = interpolate_pose(t_img, traj_times, traj_pos, traj_quat)
        line = f"{t_img:.9f} {pos[0]:.12f} {pos[1]:.12f} {pos[2]:.12f} {quat[0]:.12f} {quat[1]:.12f} {quat[2]:.12f} {quat[3]:.12f}"
        output_lines.append(line)

    with open(output_file, 'w') as f:
        f.write("\n".join(output_lines))

    print(f"Written poses for {len(output_lines)-1} images → {output_file}")

# Example usage
if __name__ == "__main__":
    synchronize_poses("traj.txt", "images", "interp_traj.txt")
