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
import socket
import netifaces
import concurrent.futures
from typing import Optional

class ESP32ObjectDetector:
    def __init__(self, capture_interval=2):
        self.capture_interval = capture_interval
        self.esp32_url = None
        self.model = YOLO('yolov8n.pt')
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.last_announced = set()
        self.output_dir = 'captured_images'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Find ESP32 camera
        self.discover_esp32()
        if not self.esp32_url:
            raise Exception("Could not find ESP32 camera on the network")
        
        cv2.namedWindow('ESP32 Camera Feed', cv2.WINDOW_NORMAL)

    def verify_esp32_camera(self, ip: str) -> Optional[str]:
        """Verify if IP hosts an ESP32 camera by checking image capture"""
        base_url = f"http://{ip}"
        try:
            # First check if server responds
            status = requests.get(f"{base_url}/status", timeout=1)
            if status.status_code != 200:
                return None
                
            # Then verify camera functionality
            capture = requests.get(f"{base_url}/capture", timeout=2)
            if capture.status_code != 200:
                return None
                
            # Verify we got an image
            try:
                Image.open(io.BytesIO(capture.content))
                print(f"Valid ESP32 camera found at: {base_url}")
                return base_url
            except:
                return None
                
        except requests.RequestException:
            return None

    def get_default_gateway_network(self) -> tuple:
        """Get default gateway IP and network range"""
        gateways = netifaces.gateways()
        if 'default' not in gateways or netifaces.AF_INET not in gateways['default']:
            raise Exception("Default gateway not found")
            
        gateway = gateways['default'][netifaces.AF_INET][0]
        base_ip = '.'.join(gateway.split('.')[:3])
        
        reverse_ip = None
        # Calculate reverse IP based on third octet
        third_octet = int(gateway.split('.')[-2])
        if(third_octet != '0' or third_octet != '1'):
            reverse_third = '1' if third_octet == 0 else '0'
            reverse_ip = f"{'.'.join(gateway.split('.')[:2])}.{reverse_third}"
        
        return (base_ip, reverse_ip)

    def discover_esp32(self):
        """Find ESP32 camera using parallel network scanning with reverse IP support"""
        base_ip, reverse_ip = self.get_default_gateway_network()
        if reverse_ip==None:
            networks = [base_ip]
        networks = [base_ip, reverse_ip]
        
        for network in networks:
            print(f"Scanning network: {network}.0/24")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
                future_to_ip = {
                    executor.submit(self.verify_esp32_camera, f"{network}.{i}"): i 
                    for i in range(1, 255)
                }
                
                for future in concurrent.futures.as_completed(future_to_ip):
                    result = future.result()
                    if result:
                        self.esp32_url = result
                        return

        raise Exception("Could not find ESP32 camera on any network")

    def capture_image(self):
        """Capture image from ESP32 camera"""
        try:
            response = requests.get(f"{self.esp32_url}/capture?_cb={int(time.time())}", timeout=5)
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error capturing image: {str(e)}")
        return None

    # Rest of the class methods remain the same
    def detect_objects(self, image):
        if image is None:
            return set(), None
        
        results = self.model(image, conf=0.5)
        detected_objects = set()
        annotated_image = image.copy()
        
        for result in results:
            for box in result.boxes:
                coords = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                
                detected_objects.add(class_name)
                
                x1, y1, x2, y2 = map(int, coords)
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(annotated_image, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return detected_objects, annotated_image

    def save_image(self, image, objects):
        if image is None or not objects:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        objects_str = "-".join(sorted(objects))
        filename = f"{timestamp}_{objects_str}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        cv2.imwrite(filepath, image)
        print(f"Saved image: {filepath}")

    def announce_objects(self, objects):
        if not objects:
            return
        
        new_objects = objects - self.last_announced
        if new_objects:
            objects_text = ", ".join(new_objects)
            print(f"Detected: {objects_text}")
            self.engine.say(f"I see {objects_text}")
            self.engine.runAndWait()
        
        self.last_announced = objects

    def run(self):
        print("Starting object detection...")
        print(f"Saving images to: {os.path.abspath(self.output_dir)}")
        try:
            while True:
                image = self.capture_image()
                if image is not None:
                    detected_objects, annotated_image = self.detect_objects(image)
                    
                    if detected_objects:
                        self.save_image(annotated_image, detected_objects)
                        self.announce_objects(detected_objects)
                    
                    cv2.imshow('ESP32 Camera Feed', annotated_image)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                time.sleep(self.capture_interval)
                
        except KeyboardInterrupt:
            print("\nStopping object detection...")
        except Exception as e:
            print(f"Error in detection loop: {str(e)}")
        finally:
            cv2.destroyAllWindows()

def main():
    detector = ESP32ObjectDetector()
    detector.run()

if __name__ == "__main__":
    main()