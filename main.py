import requests
import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
from PIL import Image
import io
import time
import os
from datetime import datetime

class ESP32ObjectDetector:
    def __init__(self, esp32_url, capture_interval=2):
        """
        Initialize the detector
        
        Args:
            esp32_url (str): Base URL of ESP32 camera (e.g., 'http://192.168.0.161')
            capture_interval (int): Seconds between captures
        """
        self.esp32_url = esp32_url.rstrip('/')
        self.capture_interval = capture_interval
        
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')  # Using the smallest YOLOv8 model for speed
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speed of speech
        
        # Keep track of last announced objects to avoid repetition
        self.last_announced = set()
        
        # Create output directory for saved images
        self.output_dir = 'captured_images'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a window for display
        cv2.namedWindow('ESP32 Camera Feed', cv2.WINDOW_NORMAL)

    def capture_image(self):
        """Capture still image from ESP32 camera"""
        try:
            # Add timestamp to prevent caching
            capture_url = f"{self.esp32_url}/capture?_cb={int(time.time())}"
            response = requests.get(capture_url, timeout=10)
            
            if response.status_code == 200:
                # Convert response content to image
                image = Image.open(io.BytesIO(response.content))
                return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                print(f"Failed to capture image. Status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error capturing image: {str(e)}")
            return None

    def detect_objects(self, image):
        """Run object detection on image and return results"""
        if image is None:
            return set(), None
        
        results = self.model(image, conf=0.5)  # Only detections with 50% or higher confidence
        detected_objects = set()
        
        # Create a copy of the image for drawing
        annotated_image = image.copy()
        
        for result in results:
            for box in result.boxes:
                # Get box coordinates and class info
                coords = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                
                # Add to detected objects set
                detected_objects.add(class_name)
                
                # Draw rectangle and label on image
                x1, y1, x2, y2 = map(int, coords)
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(annotated_image, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return detected_objects, annotated_image

    def save_image(self, image, objects):
        """Save the annotated image with timestamp and detected objects"""
        if image is None or not objects:
            return
        
        # Create filename with timestamp and detected objects
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        objects_str = "-".join(sorted(objects))
        filename = f"{timestamp}_{objects_str}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        # Save image
        cv2.imwrite(filepath, image)
        print(f"Saved image: {filepath}")

    def announce_objects(self, objects):
        """Announce detected objects using text-to-speech"""
        if not objects:
            return
        
        # Only announce new objects
        new_objects = objects - self.last_announced
        if new_objects:
            objects_text = ", ".join(new_objects)
            print(f"Detected: {objects_text}")
            self.engine.say(f"I see {objects_text}")
            self.engine.runAndWait()
        
        self.last_announced = objects

    def run(self):
        """Main detection loop"""
        print("Starting object detection...")
        print(f"Saving images to: {os.path.abspath(self.output_dir)}")
        try:
            while True:
                # Capture and process image
                image = self.capture_image()
                if image is not None:
                    detected_objects, annotated_image = self.detect_objects(image)
                    
                    if detected_objects:
                        # Save and announce detected objects
                        self.save_image(annotated_image, detected_objects)
                        self.announce_objects(detected_objects)
                    
                    # Display the image
                    cv2.imshow('ESP32 Camera Feed', annotated_image)
                    
                    # Break loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Wait before next capture
                time.sleep(self.capture_interval)
                
        except KeyboardInterrupt:
            print("\nStopping object detection...")
        except Exception as e:
            print(f"Error in detection loop: {str(e)}")
        finally:
            cv2.destroyAllWindows()

def main():
    # Replace with your ESP32's IP address
    ESP32_URL = "http://192.168.0.161"
    
    detector = ESP32ObjectDetector(ESP32_URL)
    detector.run()

if __name__ == "__main__":
    main()