import argparse
import h5py
import hdf5plugin
import progressbar
from prophesee_event_msgs.msg import EventArray, Trigger
import rosbag
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert ROS bag to HDF5 for event streams')
    parser.add_argument('input_path', type=str, help='Input ROS bag path')
    parser.add_argument('output_path', type=str, help='Output HDF5 file path')
    args = parser.parse_args()
    
    input_data_path = args.input_path
    output_data_path = args.output_path

    bag = rosbag.Bag(input_data_path, 'r')
    event_file = h5py.File(output_data_path, 'w')
    
    # Create groups for each camera
    camera1_group = event_file.create_group('/camera1_master')
    camera2_group = event_file.create_group('/camera2_slave')

    # Count total messages for progress bar
    total_messages = sum(1 for _ in bag.read_messages())
    bag.close()
    bag = rosbag.Bag(input_data_path, 'r')
    bar = progressbar.ProgressBar(maxval=total_messages).start()

    # Initialize per-camera state
    camera_event_state = {
        'camera1_master': {
            'event_curr_idx': 0,
            'event_ms_idx': 0,
            'event_ms_to_idx': [],
            'event_t_offset': None
        },
        'camera2_slave': {
            'event_curr_idx': 0,
            'event_ms_idx': 0,
            'event_ms_to_idx': [],
            'event_t_offset': None
        }
    }

    # Temporary holders for flushing
    event_x, event_y, event_p, event_t = [], [], [], []

    # Trigger holders
    trigger_timestamps = []
    trigger_polarities = []

    for i, (topic, msg, t) in enumerate(bag.read_messages(topics=[
        '/prophesee/camera1_master/cd_events_buffer',
        '/prophesee/camera2_slave/cd_events_buffer',
        '/prophesee/camera1_master/trigger_event'
    ])):
        bar.update(i + 1)

        # Map topic to camera group
        if topic.startswith('/prophesee/camera1_master'):
            current_group = camera1_group
            camera_name = 'camera1_master'
        elif topic.startswith('/prophesee/camera2_slave'):
            current_group = camera2_group
            camera_name = 'camera2_slave'
        else:
            current_group = None
            camera_name = None

        # Event messages
        if topic.endswith('cd_events_buffer'):
            state = camera_event_state[camera_name]

            if state['event_t_offset'] is None:
                state['event_t_offset'] = msg.events[0].ts

            for event in msg.events:
                t_us = int((event.ts.to_nsec() - state['event_t_offset'].to_nsec()) / 1e3)
                event_x.append(event.x)
                event_y.append(event.y)
                event_p.append(event.polarity)
                event_t.append(t_us)

                while t_us >= 1000 * state['event_ms_idx']:
                    state['event_ms_to_idx'].append(state['event_curr_idx'])
                    state['event_ms_idx'] += 1

                state['event_curr_idx'] += 1

            # Flush if chunk is large
            chunk_size = 100000
            if len(event_x) >= chunk_size:
                if 'events/t' not in current_group:
                    # Create datasets
                    current_group.create_dataset('events/x', dtype='u2', maxshape=(None,), data=np.array(event_x, dtype='u2'), **hdf5plugin.Zstd())
                    current_group.create_dataset('events/y', dtype='u2', maxshape=(None,), data=np.array(event_y, dtype='u2'), **hdf5plugin.Zstd())
                    current_group.create_dataset('events/p', dtype='u1', maxshape=(None,), data=np.array(event_p, dtype='u1'), **hdf5plugin.Zstd())
                    current_group.create_dataset('events/t', dtype='u4', maxshape=(None,), data=np.array(event_t, dtype='u4'), **hdf5plugin.Zstd())
                else:
                    # Append datasets
                    old_size = current_group['events/t'].shape[0]
                    new_size = old_size + len(event_t)
                    for name, data in zip(['x','y','p','t'], [event_x, event_y, event_p, event_t]):
                        ds = current_group[f'events/{name}']
                        ds.resize((new_size,))
                        ds[old_size:] = np.array(data, dtype=ds.dtype)
                
                # Clear temporary lists
                event_x.clear()
                event_y.clear()
                event_p.clear()
                event_t.clear()

        # Trigger messages (master camera)
        elif topic.endswith('trigger_event'):
            ts_val = msg.ts
            if hasattr(ts_val, "to_nsec"):
                ts_val = ts_val.to_nsec()
            elif isinstance(ts_val, float):
                ts_val = int(ts_val * 1e9)
            else:
                ts_val = int(ts_val)

            trigger_timestamps.append(ts_val)
            trigger_polarities.append(msg.polarity)

    # Flush remaining events for both cameras
    for camera_name, group in zip(['camera1_master','camera2_slave'], [camera1_group, camera2_group]):
        state = camera_event_state[camera_name]
        if len(event_x) > 0:
            old_size = 0
            if 'events/t' in group:
                old_size = group['events/t'].shape[0]
            new_size = old_size + len(event_t)
            for name, data in zip(['x','y','p','t'], [event_x, event_y, event_p, event_t]):
                if f'events/{name}' not in group:
                    group.create_dataset(f'events/{name}', dtype='u2' if name in ['x','y'] else ('u1' if name=='p' else 'u4'), maxshape=(None,), data=np.array(data, dtype='u2' if name in ['x','y'] else ('u1' if name=='p' else 'u4')), **hdf5plugin.Zstd())
                else:
                    ds = group[f'events/{name}']
                    ds.resize((new_size,))
                    ds[old_size:] = np.array(data, dtype=ds.dtype)
            event_x.clear()
            event_y.clear()
            event_p.clear()
            event_t.clear()

    # Store triggers (master)
    camera1_group.create_dataset('trigger/timestamps', dtype='u8', data=np.array(trigger_timestamps, dtype='u8'), **hdf5plugin.Zstd())
    camera1_group.create_dataset('trigger/polarities', dtype='u1', data=np.array(trigger_polarities, dtype='u1'), **hdf5plugin.Zstd())

    # Store ms_to_idx and t_offset for both cameras
    for camera_name, group in zip(['camera1_master','camera2_slave'], [camera1_group, camera2_group]):
        state = camera_event_state[camera_name]
        group.create_dataset('ms_to_idx', dtype='u8', data=np.array(state['event_ms_to_idx'], dtype='u8'), **hdf5plugin.Zstd())
        group.create_dataset('t_offset', shape=(1,), dtype='i8', data=int(state['event_t_offset'].to_nsec() / 1e3), **hdf5plugin.Zstd())

    bar.finish()
    event_file.close()
    bag.close()

