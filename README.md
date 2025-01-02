# 🚀 ESP32 Object Detector

## 📚 Overview
The ESP32 Object Detector is a Python-based project that captures images from an ESP32 camera, detects objects using the YOLOv8 model, and provides visual and auditory feedback. It also saves annotated images to a local directory.

## 🔧 Features
- 📸 Captures images from an ESP32 camera.
- 🎮 Detects objects in real-time using YOLOv8.
- 🔗 Saves annotated images with timestamped filenames.
- 🎤 Announces detected objects using text-to-speech.
- 🔍 Displays real-time annotated camera feed in a resizable window.

## 📝 Requirements

### 🔧 Hardware
- 🚒 ESP32 Camera module
- 💻 Computer with Python installed

### 🔧 Software
- Python 3.7 or higher

### 🔧 Python Libraries
Install the following libraries using `pip`:
```bash
pip install requests opencv-python numpy ultralytics pyttsx3 pillow
```

## 🚀 Usage

### 🔍 Setup
1. Ensure the ESP32 Camera is connected to the same network as your computer and note its IP address.
2. Replace the `ESP32_URL` variable in the `main()` function with your ESP32 Camera's IP address.

### 🔧 Running the Program
1. Run the Python script:
   ```bash
   python esp32_object_detector.py
   ```
2. Press `q` to exit the program.

### 🔗 Output
- 🕵️ Detected objects are displayed on the annotated camera feed.
- 🎤 Detected objects are announced via text-to-speech.
- 📂 Annotated images are saved in the `captured_images` directory.

## 🗋 File Structure
```
ESP32_Object_Detector/
├── esp32_object_detector.py   # Main script
├── captured_images/           # Directory for saved images
└── README.md                  # Project documentation
```

## 💡 Configuration
- **ESP32 Camera URL**: Set in the `ESP32_URL` variable in the `main()` function.
- **Capture Interval**: Adjust the `capture_interval` parameter when initializing the `ESP32ObjectDetector` class.
- **YOLO Model**: The script uses the `yolov8n.pt` model for detection. You can replace it with other YOLOv8 models for better accuracy or speed.

## 🫠 Dependencies
- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
- OpenCV
- pyttsx3 for text-to-speech
- Pillow for image processing

## 🚫 Known Issues
- ⚠️ Ensure the ESP32 Camera is powered and accessible via its IP address.
- 💡 Some YOLO models may require additional GPU support for real-time performance.

## 🎯 Future Enhancements
- 🔧 Add support for custom YOLO models.
- 🔍 Implement logging for detected objects.
- 🚀 Provide a web interface for configuration and viewing results.

## 🔒 License
This project is licensed under the MIT License. Feel free to use and modify it as per your needs.

## 🙏 Acknowledgments
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Python community for their amazing libraries and tools.

