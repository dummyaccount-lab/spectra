import rosbag
import numpy as np
import rosbag
import numpy as np
import argparse
import os
from tqdm import tqdm
import cv2
import pathlib
import shutil

# Fast drawer: vectorized & reuses a preallocated buffer
def draw_events_fast(events, width, height, frame_buf):
    # white background, reuse buffer
    frame_buf.fill(255)

    if not events:
        return frame_buf

    # build arrays from the event list
    xs  = np.fromiter((int(e.x) for e in events), dtype=np.int32, count=len(events))
    ys  = np.fromiter((int(e.y) for e in events), dtype=np.int32, count=len(events))
    pol = np.fromiter((bool(getattr(e, 'polarity', False)) for e in events),
                      dtype=np.bool_, count=len(events))

    # clip to image bounds
    m = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    if not np.any(m):
        return frame_buf
    xs, ys, pol = xs[m], ys[m], pol[m]

    # vectorized color writes (BGR)
    if np.any(pol):
        frame_buf[ys[pol], xs[pol]] = (255, 0, 0)   # positive → blue
    if np.any(~pol):
        frame_buf[ys[~pol], xs[~pol]] = (0, 0, 255) # negative → red
    return frame_buf

def main(bag_file, topics, output_dir, fps=20, width=None, height=None):
    out_dir = pathlib.Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    bag_filename = pathlib.Path(bag_file).stem
    local_out = pathlib.Path("/tmp") / f"{bag_filename}.mp4"      # write locally first
    final_out = out_dir / f"{bag_filename}.mp4"                   # then copy here

    with rosbag.Bag(bag_file, 'r') as bag:
        # only count your two topics (nicer tqdm)
        total_messages = bag.get_message_count(topic_filters=topics)
        print(f"Total messages on {topics}: {total_messages}")

        # infer width/height if not provided on CLI
        if width is None or height is None:
            first_msg_1 = next(bag.read_messages(topics=[topics[0]]))[1]
            width = int(getattr(first_msg_1, 'width'))
            height = int(getattr(first_msg_1, 'height'))
        else:
            width = int(width); height = int(height)

        # prefer FFmpeg; fallback if needed
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(local_out), cv2.CAP_FFMPEG, fourcc, fps, (width * 2, height))
        if not writer.isOpened():
            writer = cv2.VideoWriter(str(local_out), fourcc, fps, (width * 2, height))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter for {local_out}")

        # preallocate reusable buffers
        left_buf  = np.empty((height, width, 3), np.uint8)
        right_buf = np.empty((height, width, 3), np.uint8)
        combo_buf = np.empty((height, width * 2, 3), np.uint8)

        # accumulation state
        accumulated = {topic: [] for topic in topics}
        frame_duration = 1.0 / fps
        last_frame_time = None

        # iterate both topics merged by time
        for topic_name, msg, t in tqdm(bag.read_messages(topics=topics),
                                       total=total_messages, desc="Processing messages"):
            events = msg.events
            timestamp = t.to_sec()

            if last_frame_time is None:
                last_frame_time = timestamp

            # accumulate events
            accumulated[topic_name].extend(events)

            # write a frame when the window elapsed
            if (timestamp - last_frame_time) >= frame_duration:
                left_frame  = draw_events_fast(accumulated[topics[0]], width, height, left_buf)
                right_frame = draw_events_fast(accumulated[topics[1]], width, height, right_buf)

                # compose combined frame without new allocations
                combo_buf[:, :width] = left_frame
                combo_buf[:, width:] = right_frame
                writer.write(combo_buf)

                # reset & advance time window
                accumulated = {topic: [] for topic in topics}
                last_frame_time = timestamp

        writer.release()

    # copy the finished file to your chosen output directory
    shutil.copy2(local_out, final_out)
    print(f"Video saved to {final_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ROS bag events (2 topics) to a side-by-side MP4 (fast).")
    parser.add_argument('input_bag', type=str, help="Path to the input ROS bag file.")
    parser.add_argument('topics', nargs=2, type=str, help="The 2 event topics.")
    parser.add_argument('output_dir', type=str, help="Local directory to save the video (e.g., ~/Videos/ros_out).")
    parser.add_argument('--fps', type=float, default=30.0, help="Output FPS and accumulation window (default: 30).")
    parser.add_argument('--width', type=int, default=None, help="Event image width (optional override).")
    parser.add_argument('--height', type=int, default=None, help="Event image height (optional override).")
    args = parser.parse_args()

    main(args.input_bag, args.topics, args.output_dir, fps=args.fps, width=args.width, height=args.height)
