import rosbag
import argparse
from tqdm import tqdm

def filter_rosbag(input_bag_path, output_bag_path, batch_size=1000):
    # Use a set for faster membership checking
    topics_to_include = {
        '/ixblue_ins_driver/ix/ins',
        '/ixblue_ins_driver/standard/imu',
        '/ixblue_ins_driver/standard/navsatfix',
        '/ouster/imu',
        '/ouster/points',
        '/synchrobox_msg',
        '/synchrobox_pin0'
    }
    
    # Count total messages for progress tracking
    total_messages = 0
    with rosbag.Bag(input_bag_path, 'r') as input_bag:
        total_messages = input_bag.get_message_count()

    processed_messages = 0

    # Open the output rosbag file
    with rosbag.Bag(output_bag_path, 'w') as output_bag:
        # Use tqdm to create a progress bar
        with rosbag.Bag(input_bag_path, 'r') as input_bag:
            # Prepare a list for batch writing
            messages_batch = []
            for topic, msg, t in tqdm(input_bag.read_messages(), total=total_messages, desc='Processing'):
                # Check if the topic is in the set of topics to include
                if topic in topics_to_include:
                    # Collect the message in the batch
                    messages_batch.append((topic, msg, t))
                    
                    # Write in batches
                    if len(messages_batch) >= batch_size:
                        for topic_batch, msg_batch, t_batch in messages_batch:
                            output_bag.write(topic_batch, msg_batch, t_batch)
                        messages_batch.clear()  # Clear the batch after writing

                # Update processed message count
                processed_messages += 1
            
            # Write any remaining messages in the batch
            if messages_batch:
                for topic_batch, msg_batch, t_batch in messages_batch:
                    output_bag.write(topic_batch, msg_batch, t_batch)

    print("\nFiltering complete.")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Filter ROS bag files.')
    parser.add_argument('input_bag', type=str, help='Input ROS bag file path')
    parser.add_argument('-o', '--output_bag', type=str, help='Output ROS bag file path', default=None)

    args = parser.parse_args()

    # Determine output bag name based on input bag name
    if args.output_bag is None:
        base_name = args.input_bag.split('.')[0]  # Get the base name of the input bag
        args.output_bag = f'{base_name}_lidar_ins.bag'  # Append _lidar_ins to the base name

    # Call the function to filter the bag
    filter_rosbag(args.input_bag, args.output_bag)

    print(f"Filtered bag saved as: {args.output_bag}")

if __name__ == '__main__':
    main()

