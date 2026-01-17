# -*- coding: utf-8 -*-
"""
Face Blur Module
Detects faces in images and videos and applies blur only to face regions
Uses YuNet (preferred) or OpenCV DNN for face detection
"""

import cv2
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
import os
import urllib.request


class FaceBlurProcessor:
    """
    Processor for detecting and blurring faces in images and videos
    """
    
    def __init__(self, blur_intensity: int = 50):
        """
        Initialize face blur processor
        
        Args:
            blur_intensity: Blur intensity (higher = more blur, typical range: 1-50)
        """
        self.blur_intensity = blur_intensity
        self.face_detector = None
        self.dnn_net = None
        self.detection_method = None
        
        # Initialize face detector (YuNet first, then DNN)
        self._init_face_detector()
    
    def _init_face_detector(self):
        """
        Initialize face detector (優先 YuNet，回退到 DNN)
        """
        # First, try YuNet (OpenCV >= 4.5.4)
        try:
            # Check OpenCV version (need >= 4.5.4)
            cv_version_str = cv2.__version__.split('.')
            major = int(cv_version_str[0])
            minor = int(cv_version_str[1])
            patch = int(cv_version_str[2]) if len(cv_version_str) > 2 and cv_version_str[2].isdigit() else 0
            
            # Check if version >= 4.5.4
            if (major > 4) or (major == 4 and minor > 5) or (major == 4 and minor == 5 and patch >= 4):
                # Try to use YuNet - need to download model file first
                try:
                    yunet_model_path = self._download_yunet_model()
                    if yunet_model_path and os.path.exists(yunet_model_path):
                        self.face_detector = cv2.FaceDetectorYN.create(
                            model=yunet_model_path,
                            config="",
                            input_size=(320, 320),
                            score_threshold=0.5,
                            nms_threshold=0.3
                        )
                        self.detection_method = 'yunet'
                        print("Using YuNet face detector")
                        return
                    else:
                        print("YuNet model file not available, falling back to DNN")
                except Exception as e:
                    print(f"YuNet initialization failed: {str(e)}, falling back to DNN")
            else:
                print(f"OpenCV version {cv2.__version__} is too old for YuNet (need >= 4.5.4), using DNN")
        except Exception as e:
            print(f"Error checking OpenCV version: {str(e)}, trying DNN")
        
        # Fallback to DNN face detector
        try:
            pb_file, pbtxt_file = self._load_dnn_models()
            if pb_file and pbtxt_file and os.path.exists(pb_file) and os.path.exists(pbtxt_file):
                self.dnn_net = cv2.dnn.readNetFromTensorflow(pb_file, pbtxt_file)
                self.detection_method = 'dnn'
                print("Using DNN face detector")
            else:
                raise Exception("Failed to load DNN model files")
        except Exception as e:
            print(f"Warning: Failed to initialize face detector: {str(e)}")
            self.detection_method = None
    
    def _download_yunet_model(self, model_dir: str = '.') -> Optional[str]:
        """
        Download YuNet model file if it doesn't exist
        
        Args:
            model_dir: Directory to save model file
            
        Returns:
            Path to model file or None if failed
        """
        try:
            model_file = os.path.join(model_dir, 'face_detection_yunet_2023mar.onnx')
            
            if not os.path.exists(model_file):
                print("Downloading YuNet model...")
                url = 'https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
                try:
                    urllib.request.urlretrieve(url, model_file)
                    print(f"Downloaded: {model_file}")
                except Exception as e:
                    print(f"Warning: Failed to download YuNet model: {str(e)}")
                    return None
            
            return model_file if os.path.exists(model_file) else None
        except Exception as e:
            print(f"Warning: Error downloading YuNet model: {str(e)}")
            return None
    
    def _load_dnn_models(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Load DNN model files from .assets folder or current directory
        
        Returns:
            Tuple of (pb_file_path, pbtxt_file_path) or (None, None) if failed
        """
        try:
            # First try .assets folder
            assets_dir = '.assets'
            pb_file_assets = os.path.join(assets_dir, 'opencv_face_detector_uint8.pb')
            pbtxt_file_assets = os.path.join(assets_dir, 'opencv_face_detector.pbtxt')
            
            if os.path.exists(pb_file_assets) and os.path.exists(pbtxt_file_assets):
                print(f"Loading DNN models from .assets folder")
                return pb_file_assets, pbtxt_file_assets
            
            # Fallback to current directory
            pb_file = 'opencv_face_detector_uint8.pb'
            pbtxt_file = 'opencv_face_detector.pbtxt'
            
            if os.path.exists(pb_file) and os.path.exists(pbtxt_file):
                print(f"Loading DNN models from current directory")
                return pb_file, pbtxt_file
            
            # If not found, try to download
            print("DNN models not found, attempting to download...")
            return self._download_dnn_models()
            
        except Exception as e:
            print(f"Warning: Error loading DNN models: {str(e)}")
            return None, None
    
    def _download_dnn_models(self, model_dir: str = '.') -> Tuple[Optional[str], Optional[str]]:
        """
        Automatically download DNN model files if they don't exist
        
        Args:
            model_dir: Directory to save model files
            
        Returns:
            Tuple of (pb_file_path, pbtxt_file_path) or (None, None) if failed
        """
        try:
            pb_file = os.path.join(model_dir, 'opencv_face_detector_uint8.pb')
            pbtxt_file = os.path.join(model_dir, 'opencv_face_detector.pbtxt')
            
            urls = {
                pb_file: 'https://github.com/opencv/opencv/raw/4.x/samples/dnn/face_detector/opencv_face_detector_uint8.pb',
                pbtxt_file: 'https://github.com/opencv/opencv/raw/4.x/samples/dnn/face_detector/opencv_face_detector.pbtxt'
            }
            
            for filepath, url in urls.items():
                if not os.path.exists(filepath):
                    print(f"Downloading {os.path.basename(filepath)}...")
                    try:
                        urllib.request.urlretrieve(url, filepath)
                        print(f"Downloaded: {filepath}")
                    except Exception as e:
                        print(f"Warning: Failed to download {os.path.basename(filepath)}: {str(e)}")
                        return None, None
            
            return pb_file, pbtxt_file
        except Exception as e:
            print(f"Warning: Error downloading DNN models: {str(e)}")
            return None, None
    
    def detect_faces(self, img) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image
        
        Args:
            img: OpenCV image (numpy array in BGR format)
            
        Returns:
            List of face bounding boxes as (x, y, width, height) tuples
        """
        if img is None:
            return []
        
        if self.detection_method is None:
            return []
        
        faces = []
        
        try:
            if self.detection_method == 'yunet' and self.face_detector is not None:
                # Use YuNet
                h, w = img.shape[:2]
                self.face_detector.setInputSize((w, h))
                _, faces_detected = self.face_detector.detect(img)
                
                if faces_detected is not None:
                    for face in faces_detected:
                        # YuNet returns [x, y, w, h, confidence, ...]
                        x = int(face[0])
                        y = int(face[1])
                        w = int(face[2])
                        h = int(face[3])
                        confidence = face[4]
                        
                        # Lower threshold for better detection (0.3 instead of 0.5)
                        if confidence >= 0.3:  # Confidence threshold
                            faces.append((x, y, w, h))
                            print(f"Detected face (YuNet): ({x}, {y}, {w}, {h}) with confidence {confidence:.2f}")
            
            elif self.detection_method == 'dnn' and self.dnn_net is not None:
                # Use DNN
                h, w = img.shape[:2]
                blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123])
                self.dnn_net.setInput(blob)
                detections = self.dnn_net.forward()
                
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    # Lower threshold for better detection (0.3 instead of 0.5)
                    if confidence > 0.3:  # Confidence threshold
                        x1 = int(detections[0, 0, i, 3] * w)
                        y1 = int(detections[0, 0, i, 4] * h)
                        x2 = int(detections[0, 0, i, 5] * w)
                        y2 = int(detections[0, 0, i, 6] * h)
                        # Ensure valid bounding box
                        if x2 > x1 and y2 > y1:
                            faces.append((x1, y1, x2 - x1, y2 - y1))
                            print(f"Detected face: ({x1}, {y1}, {x2-x1}, {y2-y1}) with confidence {confidence:.2f}")
                        else:
                            print(f"Skipped invalid face detection: ({x1}, {y1}, {x2}, {y2})")
        
        except Exception as e:
            print(f"Warning: Face detection failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
        
        if len(faces) == 0:
            print("No faces detected in image")
        else:
            print(f"Detected {len(faces)} face(s)")
        
        return faces
    
    def blur_faces_in_image(self, img) -> Optional:
        """
        Blur only face regions in an image
        
        Args:
            img: OpenCV image (numpy array in BGR format)
            
        Returns:
            Blurred image with only faces blurred, or None if error or detector not available
        """
        if img is None:
            return None
        
        # Check if detector is available
        if self.detection_method is None:
            print("Warning: Face detector not available, cannot blur faces")
            return None
        
        try:
            # Detect faces
            faces = self.detect_faces(img)
            
            if not faces:
                # No faces detected, return original image (no blur needed)
                return img.copy()
            
            # Create a copy to work on
            blurred_img = img.copy()
            kernel_size = self.blur_intensity * 2 + 1
            
            # Blur each detected face
            for (x, y, w, h) in faces:
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, img.shape[1] - x)
                h = min(h, img.shape[0] - y)
                
                if w > 0 and h > 0:
                    # Extract face region
                    face_roi = img[y:y+h, x:x+w]
                    # Blur the face region
                    blurred_face = cv2.GaussianBlur(face_roi, (kernel_size, kernel_size), 0)
                    # Replace original face with blurred version
                    blurred_img[y:y+h, x:x+w] = blurred_face
            
            return blurred_img
            
        except Exception as e:
            print(f"Warning: Failed to blur faces in image: {str(e)}")
            return None
    
    def process_image(self, image_path: str, output_path: str) -> Optional[str]:
        """
        Process an image file: detect and blur faces, then save
        
        Args:
            image_path: Path to input image file
            output_path: Path to save output image file
            
        Returns:
            Path to saved file if successful, None otherwise
        """
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            # Blur faces
            blurred_img = self.blur_faces_in_image(img)
            if blurred_img is None:
                return None
            
            # Handle name conflicts
            output_file = Path(output_path)
            if output_file.exists():
                stem = output_file.stem
                suffix = output_file.suffix
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = str(output_file.parent / f"{stem}_{timestamp}{suffix}")
            
            # Save image
            cv2.imwrite(str(output_path), blurred_img)
            return str(output_path)
            
        except Exception as e:
            print(f"Warning: Failed to process image: {str(e)}")
            return None
    
    def process_video(self, video_path: str, output_path: str, 
                     max_frames: Optional[int] = None) -> Optional[str]:
        """
        Process a video file: detect and blur faces in each frame
        
        Args:
            video_path: Path to input video file
            output_path: Path to save output video file
            max_frames: Maximum number of frames to process (None = all frames)
            
        Returns:
            Path to saved file if successful, None otherwise
        """
        try:
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps == 0 or width == 0 or height == 0:
                cap.release()
                return None
            
            # Calculate frames to process
            if max_frames is None:
                max_frames = total_frames if total_frames > 0 else float('inf')
            else:
                max_frames = min(max_frames, total_frames) if total_frames > 0 else max_frames
            
            # Handle name conflicts
            output_file = Path(output_path)
            if output_file.exists():
                stem = output_file.stem
                suffix = output_file.suffix
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = str(output_file.parent / f"{stem}_{timestamp}{suffix}")
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            if not out.isOpened():
                cap.release()
                return None
            
            # Process frames
            kernel_size = self.blur_intensity * 2 + 1
            frame_count = 0
            
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect and blur faces in this frame
                faces = self.detect_faces(frame)
                
                if faces:
                    # Blur only face regions
                    blurred_frame = frame.copy()
                    for (x, y, w, h) in faces:
                        # Ensure coordinates are within frame bounds
                        x = max(0, x)
                        y = max(0, y)
                        w = min(w, frame.shape[1] - x)
                        h = min(h, frame.shape[0] - y)
                        
                        if w > 0 and h > 0:
                            # Extract face region
                            face_roi = frame[y:y+h, x:x+w]
                            # Blur the face region
                            blurred_face = cv2.GaussianBlur(face_roi, (kernel_size, kernel_size), 0)
                            # Replace original face with blurred version
                            blurred_frame[y:y+h, x:x+w] = blurred_face
                    
                    out.write(blurred_frame)
                else:
                    # No faces detected, write original frame
                    out.write(frame)
                
                frame_count += 1
            
            cap.release()
            out.release()
            
            return str(output_path)
            
        except Exception as e:
            print(f"Warning: Failed to process video: {str(e)}")
            return None

