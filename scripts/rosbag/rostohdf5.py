import base64
import struct
import rosbag
import h5py
import json
from tqdm import tqdm

# Function to flatten ixblue_ins_msgs/Ins message specifically
def flatten_ixblue_ins_msg(msg):
    return {
        'header': {
            'seq': msg.header.seq,
            'stamp': {
                'sec': msg.header.stamp.secs,
                'nsec': msg.header.stamp.nsecs
            },
            'frame_id': msg.header.frame_id
        },
        'ALT_REF_GEOID': msg.ALT_REF_GEOID,
        'ALT_REF_ELLIPSOID': msg.ALT_REF_ELLIPSOID,
        'latitude': msg.latitude,
        'longitude': msg.longitude,
        'altitude_ref': msg.altitude_ref,
        'altitude': msg.altitude,
        'position_covariance': msg.position_covariance,
        'heading': msg.heading,
        'roll': msg.roll,
        'pitch': msg.pitch,
        'attitude_covariance': msg.attitude_covariance,
        'speed_vessel_frame': {
            'x': msg.speed_vessel_frame.x,
            'y': msg.speed_vessel_frame.y,
            'z': msg.speed_vessel_frame.z
        },
        'speed_vessel_frame_covariance': msg.speed_vessel_frame_covariance
    }

# Function to flatten sensor_msgs/Imu message
def flatten_imu_msg(msg):
    return {
        'header': {
            'seq': msg.header.seq,
            'stamp': {
                'sec': msg.header.stamp.secs,
                'nsec': msg.header.stamp.nsecs
            },
            'frame_id': msg.header.frame_id
        },
        'orientation': {
            'x': msg.orientation.x,
            'y': msg.orientation.y,
            'z': msg.orientation.z,
            'w': msg.orientation.w
        },
        'angular_velocity': {
            'x': msg.angular_velocity.x,
            'y': msg.angular_velocity.y,
            'z': msg.angular_velocity.z
        },
        'linear_acceleration': {
            'x': msg.linear_acceleration.x,
            'y': msg.linear_acceleration.y,
            'z': msg.linear_acceleration.z
        }
    }

# Function to flatten sensor_msgs/NavSatFix message
def flatten_navsatfix_msg(msg):
    return {
        'header': {
            'seq': msg.header.seq,
            'stamp': {
                'sec': msg.header.stamp.secs,
                'nsec': msg.header.stamp.nsecs
            },
            'frame_id': msg.header.frame_id
        },
        'status': {
            'status': msg.status.status,
            'service': msg.status.service
        },
        'latitude': msg.latitude,
        'longitude': msg.longitude,
        'altitude': msg.altitude,
        'position_covariance': msg.position_covariance,
        'position_covariance_type': msg.position_covariance_type
    }

# Function to flatten prophesee_event_msgs/Event message
def flatten_event_msg(msg):
    return {
        'header': {
            'seq': msg.header.seq,
            'stamp': {
                'sec': msg.header.stamp.secs,
                'nsec': msg.header.stamp.nsecs
            },
            'frame_id': msg.header.frame_id
        },
        'events': [
            {
                'x': event.x,
                'y': event.y,
                'ts': event.ts,  # Assuming ts is of type Time; we'll convert it below
                'polarity': event.polarity
            } for event in msg.events
        ]
    }

def flatten_image_msg(msg):
    encoded_image = base64.b64encode(msg.data).decode('utf-8')
    return {
        'header': {
            'seq': msg.header.seq,
            'stamp': {
                'sec': msg.header.stamp.secs,
                'nsec': msg.header.stamp.nsecs
            },
            'frame_id': msg.header.frame_id
        },
        'encoding': msg.encoding,
        'width': msg.width,
        'height': msg.height,
        'data': encoded_image
    }

def ros_datatype_to_struct_format(datatype):
    if datatype == 1:  # ROS_FLOAT32
        return 'f'
    elif datatype == 2:  # ROS_FLOAT64
        return 'd'
    elif datatype == 3:  # ROS_UINT8
        return 'B'
    elif datatype == 4:  # ROS_INT8
        return 'b'
    elif datatype == 5:  # ROS_UINT16
        return 'H'
    elif datatype == 6:  # ROS_INT16
        return 'h'
    elif datatype == 7:  # ROS_UINT32
        return 'I'
    elif datatype == 8:  # ROS_INT32
        return 'i'
    elif datatype == 9:  # ROS_UINT64
        return 'Q'
    elif datatype == 10: # ROS_INT64
        return 'q'
    else:
        raise ValueError(f"Unsupported datatype: {datatype}")

# Function to flatten sensor_msgs/PointCloud2 message
def flatten_pointcloud2_msg(msg):
    # Initialize a list to hold the points
    points = []

    # Extract point data
    point_data = msg.data
    point_step = msg.point_step
    fields = msg.fields

    # Create a format string for struct unpacking
    field_formats = ''.join([f"{field.count}{ros_datatype_to_struct_format(field.datatype)}" for field in fields])
    field_size = struct.calcsize(field_formats)

    # Loop through the data and unpack each point
    for i in range(msg.width * msg.height):
        offset = i * point_step
        point = struct.unpack(field_formats, point_data[offset:offset + field_size])
        
        # Create a dictionary for this point
        point_dict = {}
        for j, field in enumerate(fields):
            point_dict[field.name] = point[j]
        
        points.append(point_dict)

    return {
        'header': {
            'seq': msg.header.seq,
            'stamp': {
                'sec': msg.header.stamp.secs,
                'nsec': msg.header.stamp.nsecs
            },
            'frame_id': msg.header.frame_id
        },
        'height': msg.height,
        'width': msg.width,
        'fields': [field.name for field in fields],
        'is_bigendian': msg.is_bigendian,
        'point_step': msg.point_step,
        'row_step': msg.row_step,
        'data': points,
        'is_dense': msg.is_dense
    }

# General function to convert ROS message to a dictionary
def msg_to_dict(msg):
    if isinstance(msg, dict):
        return {k: msg_to_dict(v) for k, v in msg.items()}
    elif hasattr(msg, '__slots__'):
        return {slot: msg_to_dict(getattr(msg, slot)) for slot in msg.__slots__}
    else:
        return msg

# Convert ROS bag to HDF5
def convert_rosbag_to_hdf5(rosbag_file, hdf5_file):
    with rosbag.Bag(rosbag_file, 'r') as bag:
        num_messages = bag.get_message_count()  # Get the total number of messages
        with h5py.File(hdf5_file, 'w') as h5f:
            # Wrap the message processing in a tqdm progress bar
            for topic, msg, t in tqdm(bag.read_messages(), total=num_messages, desc="Processing messages"):
                if topic == '/ixblue_ins_driver/ix/ins':
                    flattened_msg = flatten_ixblue_ins_msg(msg)
                elif topic == '/ixblue_ins_driver/standard/imu':
                    flattened_msg = flatten_imu_msg(msg)
                elif topic == '/ixblue_ins_driver/standard/navsatfix':
                    flattened_msg = flatten_navsatfix_msg(msg)
                elif topic == '/prophesee/camera2_slave/cd_events_buffer':
                    flattened_msg = flatten_event_msg(msg)
                elif topic == '/prophesee/camera1_master/cd_events_buffer':
                    flattened_msg = flatten_event_msg(msg)  # For EventArray
                elif topic == '/stereo/left/image_color':
                    flattened_msg = flatten_image_msg(msg)  # Handle image message
                elif topic == '/stereo/right/image_color':
                    flattened_msg = flatten_image_msg(msg)  # Handle image message
                elif topic == '/ouster/points':
                    flattened_msg = flatten_pointcloud2_msg(msg)
                else:
                    flattened_msg = msg_to_dict(msg)

                # Convert timestamp fields to JSON serializable format
                for event in flattened_msg.get('events', []):
                    event['ts'] = {'sec': event['ts'].secs, 'nsec': event['ts'].nsecs}  # Convert Time type to dict

                # Save message data to HDF5
                group = h5f.require_group(topic)
                json_data = json.dumps(flattened_msg)

                # Use variable-length string dtype for the dataset
                group.create_dataset(str(t.to_sec()), data=json_data, dtype=h5py.string_dtype(encoding='utf-8'))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert ROS bag to HDF5.')
    parser.add_argument('rosbag_file', type=str, help='Input rosbag file')
    parser.add_argument('hdf5_file', type=str, help='Output HDF5 file')
    args = parser.parse_args()

    convert_rosbag_to_hdf5(args.rosbag_file, args.hdf5_file)
