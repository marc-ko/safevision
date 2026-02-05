# -*- coding: utf-8 -*-
"""
Content Classifier based on "Three Point Rule" (三點不露)
Detects if content is safe according to the rule: no exposure of critical body parts
"""

import os
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

from nudenet import NudeDetector
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import shutil
from datetime import datetime
import zipfile
import numpy as np

# ============================================================================
# CONFIGURATION VARIABLES - Edit these to customize behavior
# ============================================================================

# Confidence threshold for general body part detection (0.0-1.0)
CONFIDENCE_THRESHOLD = 0.2

# Lower threshold specifically for genitalia detection (0.0-1.0)
# This allows more sensitive detection of genitalia in edge cases
GENITALIA_THRESHOLD = 0.3

# Folder to move safe images to (None = don't move)
SAFE_FOLDER = "safe_images"

# Folder to move unsafe images to (None = don't move)
UNSAFE_FOLDER = "unsafe_images"

# Folder to save censored (blurred) versions of unsafe images (None = don't censor)
CENSOR_FOLDER = "censor_images"

# Folder to save face-censored versions (None = don't create face-censored versions)
FACE_CENSOR_FOLDER = "face_censors"

# Blur intensity for censoring (higher = more blur, typical range: 1-50)
# Lower values (5-15) = light blur, Higher values (20-50) = heavy blur
BLUR_INTENSITY = 70

# Percentage of video to keep from the beginning (0.0-1.0, None = don't trim)
# Example: 0.15 means keep first 15% of the video
VIDEO_TRIM_PERCENTAGE = 0.05

# ============================================================================


class ThreePointRuleClassifier:
    """
    Classifier that determines if content is safe based on "Three Point Rule"
    Three points = Female breasts (2 points) + Genitalia (1 point)
    or Male genitalia (1 point)
    """
    
    def __init__(self, confidence_threshold: Optional[float] = None, safe_folder: Optional[str] = None, unsafe_folder: Optional[str] = None, 
                 genitalia_threshold: Optional[float] = None, censor_folder: Optional[str] = None, face_censor_folder: Optional[str] = None):
        """
        Initialize the classifier
        
        Args:
            confidence_threshold: Minimum confidence score to consider a detection valid (0.0-1.0)
                                 If None, uses CONFIDENCE_THRESHOLD from top of file (default: 0.5)
            safe_folder: Optional path to folder where safe images will be moved
                        If None, uses SAFE_FOLDER from top of file (default: None, no moving)
            unsafe_folder: Optional path to folder where unsafe images will be moved
                          If None, uses UNSAFE_FOLDER from top of file (default: None, no moving)
            genitalia_threshold: Lower threshold for genitalia detection
                                If None, uses GENITALIA_THRESHOLD from top of file
                                If still None, uses confidence_threshold * 0.6
            censor_folder: Optional path to folder where censored (blurred) versions will be saved
                          If None, uses CENSOR_FOLDER from top of file (default: None, no censoring)
            face_censor_folder: Optional path to folder where face-censored versions will be saved
                               If None, uses FACE_CENSOR_FOLDER from top of file (default: None, no face censoring)
        """
        self.detector = NudeDetector()
        
        # Use module-level variables if not provided
        if confidence_threshold is None:
            confidence_threshold = CONFIDENCE_THRESHOLD
        
        if safe_folder is None:
            safe_folder = SAFE_FOLDER
        
        if unsafe_folder is None:
            unsafe_folder = UNSAFE_FOLDER
        
        if genitalia_threshold is None:
            genitalia_threshold = GENITALIA_THRESHOLD if GENITALIA_THRESHOLD is not None else confidence_threshold * 0.6
        
        if censor_folder is None:
            censor_folder = CENSOR_FOLDER
        
        if face_censor_folder is None:
            face_censor_folder = FACE_CENSOR_FOLDER
        
        # Always use module-level variables for blur and video trim
        self.confidence_threshold = confidence_threshold
        self.genitalia_threshold = genitalia_threshold
        self.safe_folder = safe_folder
        self.unsafe_folder = unsafe_folder
        self.censor_folder = censor_folder
        self.face_censor_folder = face_censor_folder
        self.blur_intensity = BLUR_INTENSITY
        self.video_trim_percentage = VIDEO_TRIM_PERCENTAGE
        
        # Define the "three points" categories
        # Note: NudeNet uses these class names
        self.critical_points = {
            'FEMALE_BREAST_EXPOSED': {
                'name': '女性胸部',
                'name_en': 'Female Breasts',
                'points': 2
            },
            'FEMALE_GENITALIA_EXPOSED': {
                'name': '女性私處',
                'name_en': 'Female Genitalia',
                'points': 1
            },
            'MALE_GENITALIA_EXPOSED': {
                'name': '男性私處',
                'name_en': 'Male Genitalia',
                'points': 1
            },
            # Also support old naming convention for compatibility
            'EXPOSED_BREAST_F': {
                'name': '女性胸部',
                'name_en': 'Female Breasts',
                'points': 2
            },
            'EXPOSED_GENITALIA_F': {
                'name': '女性私處',
                'name_en': 'Female Genitalia',
                'points': 1
            },
            'EXPOSED_GENITALIA_M': {
                'name': '男性私處',
                'name_en': 'Male Genitalia',
                'points': 1
            }
        }
    
    def classify_image(self, image_path: str, preserve_subfolder: Optional[str] = None) -> Dict:
        """
        Classify a single image based on three point rule
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing classification results
        """
        # Check if file exists (handle Unicode paths)
        if not os.path.exists(image_path):
            return {
                'safe': False,
                'error': f'Image file not found: {image_path}',
                'detected_points': []
            }
        
        try:
            # For Unicode paths, read image first and save to temp file
            # NudeNet's detect() method may not handle Unicode paths correctly
            import cv2
            import tempfile
            
            img = self._imread_unicode(image_path)
            if img is None:
                return {
                    'safe': False,
                    'error': f'Could not read image file: {image_path}',
                    'detected_points': []
                }
            
            # Save to temporary file with ASCII path for NudeNet
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                cv2.imwrite(tmp_path, img)
            
            try:
                # Detect all body parts in the image
                results = self.detector.detect(tmp_path)
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            
            # Check for critical points
            detected_points = []
            total_points = 0
            
            for result in results:
                class_name = result.get('class', '')
                score = result.get('score', 0.0)
                
                if class_name in self.critical_points:
                    # Use lower threshold for genitalia detection
                    is_genitalia = 'GENITALIA' in class_name
                    threshold = self.genitalia_threshold if is_genitalia else self.confidence_threshold
                    
                    if score >= threshold:
                        point_info = self.critical_points[class_name].copy()
                        point_info['confidence'] = float(score)
                        point_info['bbox'] = result.get('box', [])
                        detected_points.append(point_info)
                        total_points += point_info['points']
            
            # Determine if content is safe
            is_safe = len(detected_points) == 0
            
            # Move to appropriate folder based on classification
            moved_to = None
            censored_to = None
            face_censored_to = None
            
            if is_safe and self.safe_folder:
                moved_to = self._move_to_folder(image_path, self.safe_folder, preserve_subfolder)
            elif not is_safe:
                # Create censored version first if censor_folder is set
                if self.censor_folder:
                    censored_to = self._apply_blur_censor(image_path, preserve_subfolder)
                
                # Create face-censored version if face_censor_folder is set
                if self.face_censor_folder:
                    face_censored_to = self._apply_blur_face_censor(image_path, preserve_subfolder)
                
                # Then move original to unsafe folder
                if self.unsafe_folder:
                    moved_to = self._move_to_folder(image_path, self.unsafe_folder, preserve_subfolder)
            
            return {
                'safe': is_safe,
                'image_path': image_path,
                'moved_to': moved_to,
                'censored_to': censored_to,
                'face_censored_to': face_censored_to,
                'total_points_detected': total_points,
                'detected_points': detected_points,
                'reason': '三點不露，內容安全' if is_safe else f'檢測到 {total_points} 點，不符合三點不露標準',
                'all_detections': [
                    {
                        'class': r.get('class', ''),
                        'confidence': float(r.get('score', 0.0)),
                        'box': r.get('box', [])
                    }
                    for r in results
                ]
            }
            
        except Exception as e:
            return {
                'safe': False,
                'error': f'Error processing image: {str(e)}',
                'image_path': image_path,
                'detected_points': []
            }
    
    def _move_to_folder(self, image_path: str, target_folder: str, preserve_subfolder: Optional[str] = None) -> Optional[str]:
        """
        Move image to target folder
        
        Args:
            image_path: Path to the image file
            target_folder: Target folder path
            preserve_subfolder: Optional subfolder name to preserve structure (e.g., "glass_girl")
            
        Returns:
            New path if moved successfully, None otherwise
        """
        if not target_folder:
            return None
        
        try:
            source_path = Path(image_path)
            if not source_path.exists():
                return None
            
            # Create target folder if it doesn't exist
            target_path = Path(target_folder)
            if preserve_subfolder:
                # Preserve subfolder structure
                target_path = target_path / preserve_subfolder
            target_path.mkdir(parents=True, exist_ok=True)
            
            # Get filename and handle name conflicts
            filename = source_path.name
            dest_path = target_path / filename
            
            # If file exists, add timestamp to filename
            if dest_path.exists():
                stem = source_path.stem
                suffix = source_path.suffix
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                dest_path = target_path / f"{stem}_{timestamp}{suffix}"
            
            # Move the file
            shutil.move(str(source_path), str(dest_path))
            return str(dest_path)
            
        except Exception as e:
            print(f"Warning: Failed to move {image_path} to {target_folder}: {str(e)}")
            return None
    
    def _apply_blur_censor(self, file_path: str, preserve_subfolder: Optional[str] = None) -> Optional[str]:
        """
        Apply light blur filter to censor unsafe content (images or videos)
        Uses BLUR_INTENSITY and VIDEO_TRIM_DURATION from module-level variables
        
        Args:
            file_path: Path to the image or video file
            preserve_subfolder: Optional subfolder name to preserve structure
            
        Returns:
            Path to censored file if successful, None otherwise
        """
        if not self.censor_folder:
            return None
        
        try:
            import cv2
            
            source_path = Path(file_path)
            if not source_path.exists():
                return None
            
            # Create censor folder if it doesn't exist
            censor_path = Path(self.censor_folder)
            if preserve_subfolder:
                censor_path = censor_path / preserve_subfolder
            censor_path.mkdir(parents=True, exist_ok=True)
            
            # Check if it's a video file
            video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv')
            is_video = source_path.suffix.lower() in video_extensions
            
            if is_video:
                return self._apply_blur_censor_video(file_path, preserve_subfolder)
            else:
                return self._apply_blur_censor_image(file_path, preserve_subfolder)
            
        except ImportError:
            print(f"Warning: OpenCV required for blur censoring. Install with: pip install opencv-python")
            return None
        except Exception as e:
            print(f"Warning: Failed to create censored version: {str(e)}")
            return None
    
    def _apply_blur_censor_image(self, image_path: str, preserve_subfolder: Optional[str] = None) -> Optional[str]:
        """
        Apply blur to image file
        Uses BLUR_INTENSITY from module-level variable
        
        Args:
            image_path: Path to the image file
            preserve_subfolder: Optional subfolder name to preserve structure
            
        Returns:
            Path to censored image if successful, None otherwise
        """
        try:
            import cv2
            
            source_path = Path(image_path)
            censor_path = Path(self.censor_folder)
            if preserve_subfolder:
                censor_path = censor_path / preserve_subfolder
            censor_path.mkdir(parents=True, exist_ok=True)
            
            # Read image (handle Chinese characters in path)
            img = self._imread_unicode(str(source_path))
            if img is None:
                return None
            
            # Apply Gaussian blur (light blur)
            # Kernel size must be odd, so we use blur_intensity * 2 + 1
            kernel_size = self.blur_intensity * 2 + 1
            blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            
            # Save censored image
            filename = source_path.name
            dest_path = censor_path / filename
            
            # Handle name conflicts
            if dest_path.exists():
                stem = source_path.stem
                suffix = source_path.suffix
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                dest_path = censor_path / f"{stem}_{timestamp}{suffix}"
            
            # Save image (handle Unicode paths using cv2.imencode)
            try:
                ext = dest_path.suffix.lower()
                if ext == '.jpg' or ext == '.jpeg':
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                    ext_str = '.jpg'
                elif ext == '.png':
                    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
                    ext_str = '.png'
                else:
                    encode_param = []
                    ext_str = ext if ext else '.jpg'
                
                success, encoded_img = cv2.imencode(ext_str, blurred, encode_param)
                if success:
                    with open(dest_path, 'wb') as f:
                        f.write(encoded_img.tobytes())
                else:
                    print(f"Warning: Failed to encode censored image: {dest_path.name}")
                    return None
            except Exception as e:
                print(f"Warning: Failed to save censored image {dest_path.name}: {e}")
                return None
            
            # Verify file was created
            if not dest_path.exists():
                print(f"Warning: Censored image file was not created: {dest_path}")
                return None
            
            return str(dest_path)
            
        except Exception as e:
            print(f"Warning: Failed to create censored image: {str(e)}")
            return None
    
    def _apply_blur_censor_video(self, video_path: str, preserve_subfolder: Optional[str] = None) -> Optional[str]:
        """
        Apply blur to video file and trim to first N% of frames
        Uses BLUR_INTENSITY and VIDEO_TRIM_PERCENTAGE from module-level variables
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Path to censored video if successful, None otherwise
        """
        try:
            import cv2
            
            source_path = Path(video_path)
            censor_path = Path(self.censor_folder)
            if preserve_subfolder:
                censor_path = censor_path / preserve_subfolder
            censor_path.mkdir(parents=True, exist_ok=True)
            
            # Open video (handle Chinese characters in path)
            cap = self._videocapture_unicode(str(source_path))
            if cap is None or not cap.isOpened():
                return None
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps == 0 or width == 0 or height == 0 or total_frames == 0:
                cap.release()
                print(f"Warning: Invalid video properties for censoring: {source_path.name}")
                return None
            
            # Calculate frames to process (first N% of video)
            if self.video_trim_percentage:
                max_frames = max(1, int(total_frames * self.video_trim_percentage))
            else:
                max_frames = total_frames
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            filename = source_path.name
            dest_path = censor_path / filename
            
            # Handle name conflicts
            if dest_path.exists():
                stem = source_path.stem
                suffix = source_path.suffix
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                dest_path = censor_path / f"{stem}_{timestamp}{suffix}"
            
            # Keep original FPS (video will be shorter but not sped up)
            out = cv2.VideoWriter(str(dest_path), fourcc, fps, (width, height))
            
            if not out.isOpened():
                cap.release()
                print(f"Warning: Could not create output video writer for: {source_path.name}")
                return None
            
            # Process first N% of frames
            kernel_size = self.blur_intensity * 2 + 1
            frame_count = 0
            
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply blur
                blurred_frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
                out.write(blurred_frame)
                frame_count += 1
            
            cap.release()
            out.release()
            
            # Verify output file was created
            if not dest_path.exists():
                print(f"Warning: Censored video file was not created: {dest_path}")
                return None
            
            return str(dest_path)
            
        except Exception as e:
            print(f"Warning: Failed to create censored video: {str(e)}")
            return None
    
    def _apply_blur_face_censor(self, file_path: str, preserve_subfolder: Optional[str] = None) -> Optional[str]:
        """
        Apply face-only blur filter to censor faces in images or videos
        Uses face_blur module to detect and blur only face regions
        
        Args:
            file_path: Path to the image or video file
            
        Returns:
            Path to face-censored file if successful, None otherwise
        """
        if not self.face_censor_folder:
            return None
        
        try:
            from face_blur import FaceBlurProcessor
            
            source_path = Path(file_path)
            if not source_path.exists():
                return None
            
            # Create face censor folder if it doesn't exist
            face_censor_path = Path(self.face_censor_folder)
            if preserve_subfolder:
                face_censor_path = face_censor_path / preserve_subfolder
            face_censor_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize face blur processor
            face_processor = FaceBlurProcessor(blur_intensity=self.blur_intensity)
            
            # Check if face detector is available
            if face_processor.detection_method is None:
                print("Warning: Face detector not available, skipping face censoring")
                return None
            
            # Check if it's a video file
            video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv')
            is_video = source_path.suffix.lower() in video_extensions
            
            # First, check if there are any faces in the image/video
            import cv2
            if is_video:
                # For video, check first frame for faces
                cap = self._videocapture_unicode(str(source_path))
                if cap is not None and cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        faces = face_processor.detect_faces(frame)
                        if not faces:
                            print("No faces detected in video, skipping face censoring")
                            return None
                else:
                    return None
            else:
                # For image, check if faces exist
                img = self._imread_unicode(str(source_path))
                if img is None:
                    return None
                faces = face_processor.detect_faces(img)
                if not faces:
                    print("No faces detected in image, skipping face censoring")
                    return None
            
            # Prepare output path
            filename = source_path.name
            dest_path = face_censor_path / filename
            
            # Handle name conflicts
            if dest_path.exists():
                stem = source_path.stem
                suffix = source_path.suffix
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                dest_path = face_censor_path / f"{stem}_{timestamp}{suffix}"
            
            if is_video:
                # Calculate max frames if video trim is enabled
                if self.video_trim_percentage:
                    cap = self._videocapture_unicode(str(source_path))
                    if cap is not None and cap.isOpened():
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        max_frames = max(1, int(total_frames * self.video_trim_percentage)) if total_frames > 0 else None
                        cap.release()
                    else:
                        max_frames = None
                else:
                    max_frames = None
                
                return face_processor.process_video(str(source_path), str(dest_path), max_frames=max_frames)
            else:
                return face_processor.process_image(str(source_path), str(dest_path))
            
        except ImportError:
            print(f"Warning: face_blur module required for face censoring. Make sure face_blur.py exists.")
            return None
        except Exception as e:
            print(f"Warning: Failed to create face-censored version: {str(e)}")
            return None
    
    def _imread_unicode(self, image_path: str):
        """
        Read image with Unicode path support (handles Chinese characters)
        
        Args:
            image_path: Path to image file (may contain Chinese characters)
            
        Returns:
            OpenCV image (numpy array) or None if failed
        """
        try:
            import cv2
            # Use numpy to read file bytes, then decode with OpenCV
            # This method handles Unicode paths correctly
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
                if len(image_bytes) == 0:
                    return None
                image_data = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                return img
        except Exception as e:
            # Fallback: try direct path (may work in some cases)
            try:
                import cv2
                return cv2.imread(image_path)
            except:
                return None
    
    def _videocapture_unicode(self, video_path: str):
        """
        Open video with Unicode path support (handles Chinese characters)
        
        Args:
            video_path: Path to video file (may contain Chinese characters)
            
        Returns:
            cv2.VideoCapture object or None if failed
        """
        try:
            import cv2
            # On Windows, use \\?\ prefix to handle Unicode paths
            if sys.platform == 'win32':
                # Convert to absolute path and add \\?\ prefix for long/Unicode paths
                abs_path = os.path.abspath(video_path)
                unicode_path = '\\\\?\\' + abs_path
                cap = cv2.VideoCapture(unicode_path)
            else:
                cap = cv2.VideoCapture(video_path)
            return cap
        except Exception as e:
            return None
    
    def _create_output_zip(self) -> Optional[str]:
        """
        Copy all output folders to result_pack folder 
        
        Returns:
            Path to result_pack folder if successful, None otherwise
        """
        try:
            # Define folders to include
            folders_to_copy = [
                'censor_images',
                'safe_images',
                'unsafe_images',
                'face_censors'
            ]
            
            # Check if any folder has content
            has_content = False
            for folder in folders_to_copy:
                folder_path = Path(folder)
                if folder_path.exists() and any(folder_path.iterdir()):
                    has_content = True
                    break
            
            if not has_content:
                print("No output files to copy, skipping result_pack creation")
                return None
            
            # Create result_pack folder if it doesn't exist
            result_pack_dir = Path('result_pack')
            result_pack_dir.mkdir(parents=True, exist_ok=True)
            
            # Create timestamped subfolder
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_folder = result_pack_dir / f"classification_results_{timestamp}"
            output_folder.mkdir(parents=True, exist_ok=True)
            
            print(f"\nCopying output folders to: {output_folder}")
            
            # Copy each folder to result_pack, preserving structure
            for folder_name in folders_to_copy:
                source_folder = Path(folder_name)
                if source_folder.exists():
                    dest_folder = output_folder / folder_name
                    # Copy entire folder structure
                    if source_folder.is_dir():
                        shutil.copytree(source_folder, dest_folder, dirs_exist_ok=True)
                        print(f"  Copied: {folder_name}/")
            
            print(f"Output folder created: {output_folder}")
            return str(output_folder)
            
        except Exception as e:
            print(f"Warning: Failed to create result_pack folder: {str(e)}")
            return None
    
    def _save_results_append(self, output_file: str, new_results):
        """
        Append results to existing JSON file or create new one
        
        Args:
            output_file: Path to JSON file
            new_results: New results to append (can be dict for batch or single result)
        """
        output_path = Path(output_file)
        
        # If file exists, load existing data
        if output_path.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                # Check if new_results is batch format (has 'results' key)
                is_batch = isinstance(new_results, dict) and 'results' in new_results
                
                # Merge results
                if isinstance(existing_data, dict) and 'results' in existing_data:
                    # Existing data is batch format
                    if is_batch:
                        # Both are batch format, merge counts and results
                        existing_data['total_images'] = existing_data.get('total_images', 0) + new_results.get('total_images', 0)
                        existing_data['safe_count'] = existing_data.get('safe_count', 0) + new_results.get('safe_count', 0)
                        existing_data['unsafe_count'] = existing_data.get('unsafe_count', 0) + new_results.get('unsafe_count', 0)
                        existing_data['error_count'] = existing_data.get('error_count', 0) + new_results.get('error_count', 0)
                        existing_data['results'].extend(new_results.get('results', []))
                    else:
                        # Existing is batch, new is single result
                        existing_data['total_images'] = existing_data.get('total_images', 0) + 1
                        if 'error' in new_results:
                            existing_data['error_count'] = existing_data.get('error_count', 0) + 1
                        elif new_results.get('safe', False):
                            existing_data['safe_count'] = existing_data.get('safe_count', 0) + 1
                        else:
                            existing_data['unsafe_count'] = existing_data.get('unsafe_count', 0) + 1
                        existing_data['results'].append(new_results)
                    merged_data = existing_data
                elif isinstance(existing_data, list):
                    # Existing data is list format
                    if is_batch:
                        existing_data.extend(new_results.get('results', []))
                    else:
                        existing_data.append(new_results)
                    merged_data = existing_data
                else:
                    # Existing data is single result, convert to batch format
                    if is_batch:
                        # Convert existing to batch and merge
                        merged_data = {
                            'total_images': 1 + new_results.get('total_images', 0),
                            'safe_count': (1 if existing_data.get('safe', False) else 0) + new_results.get('safe_count', 0),
                            'unsafe_count': (0 if existing_data.get('safe', False) else 1) + new_results.get('unsafe_count', 0),
                            'error_count': (1 if 'error' in existing_data else 0) + new_results.get('error_count', 0),
                            'results': [existing_data] + new_results.get('results', [])
                        }
                    else:
                        # Both are single results, convert to list
                        merged_data = [existing_data, new_results]
            except (json.JSONDecodeError, IOError) as e:
                # If file is corrupted or can't be read, create new
                print(f"Warning: Could not read existing file, creating new: {str(e)}")
                merged_data = new_results
        else:
            # File doesn't exist, use new results as is
            merged_data = new_results
        
        # Write merged data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    def classify_batch(self, image_paths: List[str], output_file: Optional[str] = None) -> Dict:
        """
        Classify multiple images and videos
        
        Args:
            image_paths: List of image/video file paths
            output_file: Optional path to save results as JSON
            
        Returns:
            Dictionary containing results for all files
        """
        # Separate images and videos
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv')
        image_files = []
        video_files = []
        
        for file_path in image_paths:
            path = Path(file_path)
            if path.suffix.lower() in video_extensions:
                video_files.append(file_path)
            else:
                image_files.append(file_path)
        
        results = {
            'total_images': len(image_files),
            'total_videos': len(video_files),
            'safe_count': 0,
            'unsafe_count': 0,
            'error_count': 0,
            'results': []
        }
        
        # Process images
        for image_path in image_files:
            classification = self.classify_image(image_path)
            results['results'].append(classification)
            
            if 'error' in classification:
                results['error_count'] += 1
            elif classification['safe']:
                results['safe_count'] += 1
            else:
                results['unsafe_count'] += 1
        
        # Process videos
        for video_path in video_files:
            classification = self.classify_video(video_path)
            results['results'].append(classification)
            
            if 'error' in classification:
                results['error_count'] += 1
            elif classification['safe']:
                results['safe_count'] += 1
            else:
                results['unsafe_count'] += 1
        
        # Save to file if specified (append mode)
        if output_file:
            self._save_results_append(output_file, results)
            print(f"Results saved to: {output_file}")
        
        # Copy output folders to result_pack after classification
        self._create_output_zip()
        
        return results
    
    def classify_video(self, video_path: str, sample_frames: int = 10, preserve_subfolder: Optional[str] = None) -> Dict:
        """
        Classify a video file by sampling frames
        
        Args:
            video_path: Path to the video file
            sample_frames: Number of frames to sample from the video (default: 10)
            
        Returns:
            Dictionary containing classification results
        """
        # Check if file exists (handle Unicode paths)
        if not os.path.exists(video_path):
            return {
                'safe': False,
                'error': f'Video file not found: {video_path}',
                'detected_points': []
            }
        
        try:
            import cv2
            
            # Open video file
            cap = self._videocapture_unicode(video_path)
            if cap is None or not cap.isOpened():
                return {
                    'safe': False,
                    'error': f'Could not open video file: {video_path}',
                    'detected_points': []
                }
            
            # Get total frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames == 0:
                cap.release()
                return {
                    'safe': False,
                    'error': f'Video file has no frames: {video_path}',
                    'detected_points': []
                }
            
            # Sample frames evenly throughout the video
            frame_indices = [int(i * total_frames / (sample_frames + 1)) for i in range(1, sample_frames + 1)]
            
            all_detected_points = []
            max_total_points = 0
            frame_results = []
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Save frame temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    cv2.imwrite(tmp_path, frame)
                
                try:
                    # Classify this frame
                    frame_result = self.detector.detect(tmp_path)
                    
                    # Check for critical points in this frame
                    frame_detected_points = []
                    frame_total_points = 0
                    
                    for result in frame_result:
                        class_name = result.get('class', '')
                        score = result.get('score', 0.0)
                        
                        if class_name in self.critical_points:
                            is_genitalia = 'GENITALIA' in class_name
                            threshold = self.genitalia_threshold if is_genitalia else self.confidence_threshold
                            
                            if score >= threshold:
                                point_info = self.critical_points[class_name].copy()
                                point_info['confidence'] = float(score)
                                point_info['bbox'] = result.get('box', [])
                                point_info['frame'] = frame_idx
                                frame_detected_points.append(point_info)
                                frame_total_points += point_info['points']
                    
                    if frame_total_points > max_total_points:
                        max_total_points = frame_total_points
                        all_detected_points = frame_detected_points
                    
                    frame_results.append({
                        'frame': frame_idx,
                        'detected_points': len(frame_detected_points),
                        'total_points': frame_total_points
                    })
                
                finally:
                    # Clean up temp file
                    if Path(tmp_path).exists():
                        os.unlink(tmp_path)
            
            cap.release()
            
            # Determine if content is safe (unsafe if any frame has critical points)
            is_safe = max_total_points == 0
            
            # Move to appropriate folder based on classification
            moved_to = None
            censored_to = None
            face_censored_to = None
            
            if is_safe and self.safe_folder:
                moved_to = self._move_to_folder(video_path, self.safe_folder, preserve_subfolder)
            elif not is_safe:
                # Create censored version first if censor_folder is set
                if self.censor_folder:
                    censored_to = self._apply_blur_censor(video_path, preserve_subfolder)
                
                # Create face-censored version if face_censor_folder is set
                if self.face_censor_folder:
                    face_censored_to = self._apply_blur_face_censor(video_path, preserve_subfolder)
                
                # Then move original to unsafe folder
                if self.unsafe_folder:
                    moved_to = self._move_to_folder(video_path, self.unsafe_folder, preserve_subfolder)
            
            return {
                'safe': is_safe,
                'video_path': video_path,
                'total_frames': total_frames,
                'fps': fps,
                'sampled_frames': len(frame_results),
                'moved_to': moved_to,
                'censored_to': censored_to,
                'face_censored_to': face_censored_to,
                'total_points_detected': max_total_points,
                'detected_points': all_detected_points,
                'reason': '三點不露，內容安全' if is_safe else f'檢測到 {max_total_points} 點，不符合三點不露標準',
                'frame_results': frame_results
            }
            
        except ImportError:
            return {
                'safe': False,
                'error': 'OpenCV (cv2) is required for video processing. Install with: pip install opencv-python',
                'video_path': video_path,
                'detected_points': []
            }
        except Exception as e:
            return {
                'safe': False,
                'error': f'Error processing video: {str(e)}',
                'video_path': video_path,
                'detected_points': []
            }
    
    def classify_directory(self, directory_path: str, output_file: Optional[str] = None, 
                          extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.mp4', '.avi', '.mov', '.mkv')) -> Dict:
        """
        Classify all images and videos in a directory
        
        Args:
            directory_path: Path to directory containing images and videos
            output_file: Optional path to save results as JSON
            extensions: Tuple of file extensions to process
            
        Returns:
            Dictionary containing results for all files
        """
        dir_path = Path(directory_path)
        if not dir_path.exists():
            return {
                'error': f'Directory not found: {directory_path}',
                'results': []
            }
        
        # Find all files
        file_paths = []
        for ext in extensions:
            file_paths.extend(dir_path.glob(f'*{ext}'))
            file_paths.extend(dir_path.glob(f'*{ext.upper()}'))
        
        file_paths = [str(p) for p in file_paths]
        
        # Count images and videos
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv')
        image_count = sum(1 for p in file_paths if Path(p).suffix.lower() not in video_extensions)
        video_count = sum(1 for p in file_paths if Path(p).suffix.lower() in video_extensions)
        
        if image_count > 0 and video_count > 0:
            print(f"Found {image_count} images and {video_count} videos in {directory_path}")
        elif image_count > 0:
            print(f"Found {image_count} images in {directory_path}")
        elif video_count > 0:
            print(f"Found {video_count} videos in {directory_path}")
        else:
            print(f"No files found in {directory_path}")
        
        results = self.classify_batch(file_paths, output_file)
        
        # Note: result_pack folder creation is already handled in classify_batch
        
        return results
    
    def classify_untagged_folders(self, untagged_dir: str = 'untagged', output_file: Optional[str] = None) -> Dict:
        """
        Classify all subfolders in untagged directory, one by one
        Each subfolder's results are organized in output folders with subfolder structure
        Each subfolder gets its own output folder in result_pack
        
        Args:
            untagged_dir: Path to untagged directory (default: 'untagged')
            output_file: Optional path to save results as JSON
            
        Returns:
            Dictionary containing results for all subfolders
        """
        untagged_path = Path(untagged_dir)
        if not untagged_path.exists():
            return {
                'error': f'Untagged directory not found: {untagged_dir}',
                'results': []
            }
        
        # Find all subdirectories
        subfolders = [d for d in untagged_path.iterdir() if d.is_dir()]
        
        if not subfolders:
            print(f"No subfolders found in {untagged_dir}")
            return {
                'total_folders': 0,
                'results': []
            }
        
        print(f"Found {len(subfolders)} subfolders in {untagged_dir}")
        print("=" * 60)
        
        all_results = {
            'total_folders': len(subfolders),
            'folders': []
        }
        
        for subfolder in subfolders:
            subfolder_name = subfolder.name
            print(f"\nProcessing subfolder: {subfolder_name}")
            print("-" * 60)
            
            # Classify this subfolder with preserve_subfolder
            # We need to modify classify_batch to support preserve_subfolder
            # For now, let's use classify_directory and then manually organize
            
            # Find all files in subfolder
            extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.mp4', '.avi', '.mov', '.mkv')
            file_paths = []
            for ext in extensions:
                file_paths.extend(subfolder.glob(f'*{ext}'))
                file_paths.extend(subfolder.glob(f'*{ext.upper()}'))
            
            if not file_paths:
                print(f"  No files found in {subfolder_name}, skipping")
                continue
            
            # Process each file with preserve_subfolder
            folder_results = {
                'subfolder': subfolder_name,
                'total_files': len(file_paths),
                'safe_count': 0,
                'unsafe_count': 0,
                'error_count': 0,
                'results': []
            }
            
            video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv')
            for file_path in file_paths:
                is_video = file_path.suffix.lower() in video_extensions
                try:
                    if is_video:
                        result = self.classify_video(str(file_path), preserve_subfolder=subfolder_name)
                    else:
                        result = self.classify_image(str(file_path), preserve_subfolder=subfolder_name)
                    
                    folder_results['results'].append(result)
                    
                    if 'error' in result:
                        folder_results['error_count'] += 1
                    elif result.get('safe', False):
                        folder_results['safe_count'] += 1
                    else:
                        folder_results['unsafe_count'] += 1
                except Exception as e:
                    folder_results['error_count'] += 1
                    folder_results['results'].append({
                        'error': str(e),
                        'file_path': str(file_path)
                    })
            
            all_results['folders'].append(folder_results)
            
            # Create output folder for this subfolder
            output_path = self._create_output_zip_for_subfolder(subfolder_name)
            if output_path:
                print(f"  Created folder: {output_path}")
            
            print(f"  Completed: {subfolder_name} - Safe: {folder_results['safe_count']}, Unsafe: {folder_results['unsafe_count']}, Errors: {folder_results['error_count']}")
        
        # Save results if specified
        if output_file:
            self._save_results_append(output_file, all_results)
        
        print("\n" + "=" * 60)
        print(f"All subfolders processed: {len(subfolders)} folders")
        
        return all_results
    
    def _create_output_zip_for_subfolder(self, subfolder_name: str) -> Optional[str]:
        """
        Copy a specific subfolder's output to result_pack folder 
        
        Args:
            subfolder_name: Name of the subfolder (e.g., "glass_girl")
            
        Returns:
            Path to created folder if successful, None otherwise
        """
        try:
            # Define folders to include
            folders_to_copy = [
                'censor_images',
                'safe_images',
                'unsafe_images',
                'face_censors'
            ]
            
            # Check if subfolder has content in any output folder
            has_content = False
            for folder in folders_to_copy:
                folder_path = Path(folder) / subfolder_name
                if folder_path.exists() and any(folder_path.iterdir()):
                    has_content = True
                    break
            
            if not has_content:
                return None
            
            # Create result_pack folder if it doesn't exist
            result_pack_dir = Path('result_pack')
            result_pack_dir.mkdir(parents=True, exist_ok=True)
            
            # Create output folder for this subfolder
            output_folder = result_pack_dir / subfolder_name
            output_folder.mkdir(parents=True, exist_ok=True)
            
            print(f"  Creating output folder: {output_folder}")
            
            # Copy each folder's subfolder content, preserving structure
            for folder_name in folders_to_copy:
                source_subfolder = Path(folder_name) / subfolder_name
                if source_subfolder.exists():
                    dest_folder = output_folder / folder_name
                    # Copy entire subfolder structure
                    if source_subfolder.is_dir():
                        shutil.copytree(source_subfolder, dest_folder, dirs_exist_ok=True)
                        print(f"    Copied: {folder_name}/{subfolder_name}/")
            
            return str(output_folder)
            
        except Exception as e:
            print(f"  Warning: Failed to create result_pack folder for {subfolder_name}: {str(e)}")
            return None


def main():
    """
    Example usage of the classifier
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python content_classifier.py <image_path>")
        print("  python content_classifier.py <directory_path> --batch")
        print("\nExample:")
        print("  python content_classifier.py image.jpg")
        print("  python content_classifier.py ./images --batch")
        sys.exit(1)
    
    input_path = sys.argv[1]
    is_batch = '--batch' in sys.argv or '--dir' in sys.argv
    
    # Initialize classifier with default threshold
    classifier = ThreePointRuleClassifier(confidence_threshold=0.5)
    
    if is_batch or Path(input_path).is_dir():
        # Process directory
        results = classifier.classify_directory(input_path, output_file='classification_results.json')
        
        print(f"\n=== Classification Summary ===")
        print(f"Total images: {results.get('total_images', 0)}")
        print(f"Safe (三點不露): {results.get('safe_count', 0)}")
        print(f"Unsafe (檢測到三點): {results.get('unsafe_count', 0)}")
        print(f"Errors: {results.get('error_count', 0)}")
    else:
        # Process single image
        result = classifier.classify_image(input_path)
        
        print(f"\n=== Classification Result ===")
        print(f"Image: {result.get('image_path', 'N/A')}")
        print(f"Safe: {'是 (三點不露)' if result['safe'] else '否 (檢測到三點)'}")
        
        if result.get('detected_points'):
            print(f"\n檢測到的部位:")
            for point in result['detected_points']:
                print(f"  - {point['name']} ({point['name_en']})")
                print(f"    置信度: {point['confidence']:.2%}")
                print(f"    點數: {point['points']}")
        
        if result.get('error'):
            print(f"Error: {result['error']}")


if __name__ == '__main__':
    main()

