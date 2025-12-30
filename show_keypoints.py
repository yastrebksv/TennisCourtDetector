import cv2
import json
import os
import argparse
import glob

def get_contrasting_colors(n=15):
    """Generate n contrasting colors for keypoint visualization"""
    colors = [
        (255, 0, 0),     # Blue
        (0, 255, 0),     # Green
        (0, 0, 255),     # Red
        (255, 255, 0),   # Cyan
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Yellow
        (128, 0, 0),     # Dark Blue
        (0, 128, 0),     # Dark Green
        (0, 0, 128),     # Dark Red
        (128, 128, 0),   # Dark Cyan
        (128, 0, 128),   # Dark Magenta
        (0, 128, 128),   # Dark Yellow
        (192, 192, 192), # Light Gray
        (128, 128, 128), # Gray
        (255, 165, 0),   # Orange
    ]
    return colors[:n]

class KeypointVisualizer:
    def __init__(self, images_dir, json_path):
        self.images_dir = images_dir
        self.json_path = json_path
        
        # Load annotations
        with open(json_path, 'r') as f:
            self.annotations = json.load(f)
        
        # Get image paths
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        self.image_paths = []
        for ext in image_extensions:
            self.image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
            self.image_paths.extend(glob.glob(os.path.join(images_dir, ext.upper())))
        
        self.image_paths.sort()
        
        # Get corresponding annotations for each image
        self.image_annotations = {}
        for ann in self.annotations:
            # Create a mapping from image ID to annotation
            # We assume the image ID corresponds to a part of the filename
            for img_path in self.image_paths:
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                if ann['id'] in img_name or img_name in ann['id']:
                    self.image_annotations[img_name] = ann
                    break
        
        self.current_idx = 0
        self.colors = get_contrasting_colors(15)
        
        # If no direct mapping found, try to match by index
        if len(self.image_annotations) == 0:
            for i, img_path in enumerate(self.image_paths):
                if i < len(self.annotations):
                    img_name = os.path.splitext(os.path.basename(img_path))[0]
                    self.image_annotations[img_name] = self.annotations[i]
    
    def draw_keypoints(self, image, keypoints):
        """Draw keypoints on image"""
        img_with_kps = image.copy()
        
        for i, kp in enumerate(keypoints):
            x, y = int(kp[0]), int(kp[1])
            color = self.colors[i % len(self.colors)]
            
            # Draw keypoint
            cv2.circle(img_with_kps, (x, y), 5, color, -1)
            
            # Draw keypoint number
            cv2.putText(img_with_kps, str(i), (x + 10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return img_with_kps
    
    def get_current_image_with_annotations(self):
        """Get current image with keypoints overlay"""
        if self.current_idx >= len(self.image_paths):
            return None
        
        img_path = self.image_paths[self.current_idx]
        image = cv2.imread(img_path)
        
        if image is None:
            return None
        
        # Get corresponding annotation
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        annotation = self.image_annotations.get(img_name)
        
        if annotation:
            keypoints = annotation['kps']
            image_with_kps = self.draw_keypoints(image, keypoints)
            
            # Add image info
            h, w = image_with_kps.shape[:2]
            cv2.putText(image_with_kps, f"Image: {os.path.basename(img_path)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image_with_kps, f"Image: {os.path.basename(img_path)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            cv2.putText(image_with_kps, f"Index: {self.current_idx + 1}/{len(self.image_paths)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image_with_kps, f"Index: {self.current_idx + 1}/{len(self.image_paths)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            # Add navigation instructions
            cv2.putText(image_with_kps, "Press N for Next, P for Previous, Q to Quit", 
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image_with_kps, "Press N for Next, P for Previous, Q to Quit", 
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            return image_with_kps
        else:
            # If no annotation for this image, just show the image with basic info
            cv2.putText(image, f"Image: {os.path.basename(img_path)} (No annotation)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, f"Index: {self.current_idx + 1}/{len(self.image_paths)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return image
    
    def next_image(self):
        """Show next image"""
        if self.current_idx < len(self.image_paths) - 1:
            self.current_idx += 1
    
    def prev_image(self):
        """Show previous image"""
        if self.current_idx > 0:
            self.current_idx -= 1
    
    def run(self):
        """Run visualization"""
        if len(self.image_paths) == 0:
            print(f"No images found in {self.images_dir}")
            return
        
        if len(self.annotations) == 0:
            print(f"No annotations found in {self.json_path}")
            return
        
        print(f"Found {len(self.image_paths)} images and {len(self.annotations)} annotations")
        print("Controls: N - Next, P - Previous, Q - Quit")
        
        while True:
            img_with_annotations = self.get_current_image_with_annotations()
            
            if img_with_annotations is None:
                print(f"Could not load image: {self.image_paths[self.current_idx]}")
                break
            
            cv2.imshow("Keypoint Visualization", img_with_annotations)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('n') or key == ord('N'):
                self.next_image()
            elif key == ord('p') or key == ord('P'):
                self.prev_image()
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize keypoints from JSON annotations")
    parser.add_argument("--images_dir", type=str, required=True, 
                        help="Directory containing images")
    parser.add_argument("--json_path", type=str, required=True, 
                        help="Path to JSON annotations file")
    
    args = parser.parse_args()
    
    visualizer = KeypointVisualizer(args.images_dir, args.json_path)
    visualizer.run()