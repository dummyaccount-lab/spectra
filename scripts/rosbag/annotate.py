import os
import argparse

from ultralytics import YOLOv10
import cv2
import numpy as np

def load_class_names(model):
    """
    Load the class names from the model's metadata.
    
    Args:
        model: The YOLO model.
    
    Returns:
        List of class names.
    """
    return model.names

def generate_color(class_id):
    """
    Generate a unique color for each class based on the class ID.
    
    Args:
        class_id (int): The class ID for which to generate the color.
    
    Returns:
        Tuple: An (R, G, B) color for the class.
    """
    np.random.seed(class_id)  # Ensure consistent color for the same class ID
    return tuple(np.random.randint(0, 256, 3).tolist())  # Random RGB color

def auto_annotate_yolo(image_dir, output_dir, model_path='yolov10x.pt', show_probabilities=False):
    """
    Automatically annotates images using a YOLOv8 pre-trained model.
    
    Args:
        image_dir (str): Path to the folder containing images.
        output_dir (str): Path to the folder to save annotated images and labels.
        model_path (str): Path to the YOLOv8 model. Default is 'yolov8.pt'.
        show_probabilities (bool): Whether to display probability along with class name. Default is False.
    """
    # Ensure output directories exist
    annotated_dir = os.path.join(output_dir, 'annotated_images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    model = YOLOv10.from_pretrained('jameslahm/yolov10x')
    model = YOLOv10('../../../yolov10x.pt')  # Load the latest YOLO model
    
    # Load class names
    class_names = load_class_names(model)
    
    # Process each image in the directory
    for image_file in os.listdir(image_dir):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(image_dir, image_file)
            print(f"Processing: {image_path}")
            
            # Run YOLO detection
            results = model(image_path)
            
            # Get the image using OpenCV (so we can manipulate it)
            img = cv2.imread(image_path)
            
            # Draw bounding boxes and class names on the image
            for box in results[0].boxes:
                x_center, y_center, width, height = box.xywhn.tolist()[0]
                x1, y1 = int((x_center - width / 2) * img.shape[1]), int((y_center - height / 2) * img.shape[0])
                x2, y2 = int((x_center + width / 2) * img.shape[1]), int((y_center + height / 2) * img.shape[0])
                
                # Get the class name
                class_id = int(box.cls)
                class_name = class_names[class_id]
                
                # Generate a unique color for the class
                color = generate_color(class_id)
                
                # Optionally, include the probability
                if show_probabilities:
                    probability = box.conf
                    label = f"{class_name} {probability:.2f}"
                else:
                    label = class_name
                
                # Draw the bounding box and label
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Save the annotated image
            annotated_image_path = os.path.join(annotated_dir, image_file)
            cv2.imwrite(annotated_image_path, img)
            
            # Save labels in YOLO format with class names
            label_name = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_name)
            with open(label_path, 'w') as label_file:
                for box in results[0].boxes:
                    x_center, y_center, width, height = box.xywhn.tolist()[0]
                    class_id = int(box.cls)
                    class_name = class_names[class_id]
                    if len(box.xywhn.tolist()[0]) == 4:
                        label_file.write(f"{class_name} {x_center} {y_center} {width} {height}\n")
                    else:
                        print("Skipping box due to unexpected format.")
    
    print("Annotation completed!")
    print(f"Annotated images saved in: {annotated_dir}")
    print(f"Labels saved in: {labels_dir}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Auto-annotate images using YOLOv8.")
    parser.add_argument('image_dir', type=str, help="Path to the folder containing images.")
    parser.add_argument('output_dir', type=str, help="Path to the folder to save annotations.")
    parser.add_argument('--model_path', type=str, help="Path to the YOLOv8 model (default: yolov8.pt).")
    parser.add_argument('--show_probabilities', action='store_true', help="Whether to display probability in image annotations.")
    
    args = parser.parse_args()
    
    # Run auto-annotation
    auto_annotate_yolo(args.image_dir, args.output_dir, args.model_path, args.show_probabilities)

if __name__ == "__main__":
    main()

