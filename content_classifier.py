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
BLUR_INTENSITY = 50

# Percentage of video to keep from the beginning (0.0-1.0, None = don't trim)
# Example: 0.15 means keep first 15% of the video
VIDEO_TRIM_PERCENTAGE = 0.15

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
    
    def classify_image(self, image_path: str) -> Dict:
        """
        Classify a single image based on three point rule
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing classification results
        """
        if not Path(image_path).exists():
            return {
                'safe': False,
                'error': f'Image file not found: {image_path}',
                'detected_points': []
            }
        
        try:
            # Detect all body parts in the image
            results = self.detector.detect(image_path)
            
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
                moved_to = self._move_to_folder(image_path, self.safe_folder)
            elif not is_safe:
                # Create censored version first if censor_folder is set
                if self.censor_folder:
                    censored_to = self._apply_blur_censor(image_path)
                
                # Create face-censored version if face_censor_folder is set
                if self.face_censor_folder:
                    face_censored_to = self._apply_blur_face_censor(image_path)
                
                # Then move original to unsafe folder
                if self.unsafe_folder:
                    moved_to = self._move_to_folder(image_path, self.unsafe_folder)
            
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
    
    def _move_to_folder(self, image_path: str, target_folder: str) -> Optional[str]:
        """
        Move image to target folder
        
        Args:
            image_path: Path to the image file
            target_folder: Target folder path
            
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
    
    def _apply_blur_censor(self, file_path: str) -> Optional[str]:
        """
        Apply light blur filter to censor unsafe content (images or videos)
        Uses BLUR_INTENSITY and VIDEO_TRIM_DURATION from module-level variables
        
        Args:
            file_path: Path to the image or video file
            
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
            censor_path.mkdir(parents=True, exist_ok=True)
            
            # Check if it's a video file
            video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv')
            is_video = source_path.suffix.lower() in video_extensions
            
            if is_video:
                return self._apply_blur_censor_video(file_path)
            else:
                return self._apply_blur_censor_image(file_path)
            
        except ImportError:
            print(f"Warning: OpenCV required for blur censoring. Install with: pip install opencv-python")
            return None
        except Exception as e:
            print(f"Warning: Failed to create censored version: {str(e)}")
            return None
    
    def _apply_blur_censor_image(self, image_path: str) -> Optional[str]:
        """
        Apply blur to image file
        Uses BLUR_INTENSITY from module-level variable
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Path to censored image if successful, None otherwise
        """
        try:
            import cv2
            
            source_path = Path(image_path)
            censor_path = Path(self.censor_folder)
            
            # Read image
            img = cv2.imread(str(source_path))
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
            
            cv2.imwrite(str(dest_path), blurred)
            return str(dest_path)
            
        except Exception as e:
            print(f"Warning: Failed to create censored image: {str(e)}")
            return None
    
    def _apply_blur_censor_video(self, video_path: str) -> Optional[str]:
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
            
            # Open video
            cap = cv2.VideoCapture(str(source_path))
            if not cap.isOpened():
                return None
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps == 0 or width == 0 or height == 0 or total_frames == 0:
                cap.release()
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
            
            return str(dest_path)
            
        except Exception as e:
            print(f"Warning: Failed to create censored video: {str(e)}")
            return None
    
    def _apply_blur_face_censor(self, file_path: str) -> Optional[str]:
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
                cap = cv2.VideoCapture(str(source_path))
                if cap.isOpened():
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
                img = cv2.imread(str(source_path))
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
                    cap = cv2.VideoCapture(str(source_path))
                    if cap.isOpened():
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
        Classify multiple images
        
        Args:
            image_paths: List of image file paths
            output_file: Optional path to save results as JSON
            
        Returns:
            Dictionary containing results for all images
        """
        results = {
            'total_images': len(image_paths),
            'safe_count': 0,
            'unsafe_count': 0,
            'error_count': 0,
            'results': []
        }
        
        for image_path in image_paths:
            classification = self.classify_image(image_path)
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
        
        return results
    
    def classify_video(self, video_path: str, sample_frames: int = 10) -> Dict:
        """
        Classify a video file by sampling frames
        
        Args:
            video_path: Path to the video file
            sample_frames: Number of frames to sample from the video (default: 10)
            
        Returns:
            Dictionary containing classification results
        """
        if not Path(video_path).exists():
            return {
                'safe': False,
                'error': f'Video file not found: {video_path}',
                'detected_points': []
            }
        
        try:
            import cv2
            
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
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
                moved_to = self._move_to_folder(video_path, self.safe_folder)
            elif not is_safe:
                # Create censored version first if censor_folder is set
                if self.censor_folder:
                    censored_to = self._apply_blur_censor(video_path)
                
                # Create face-censored version if face_censor_folder is set
                if self.face_censor_folder:
                    face_censored_to = self._apply_blur_face_censor(video_path)
                
                # Then move original to unsafe folder
                if self.unsafe_folder:
                    moved_to = self._move_to_folder(video_path, self.unsafe_folder)
            
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
        Classify all images in a directory
        
        Args:
            directory_path: Path to directory containing images
            output_file: Optional path to save results as JSON
            extensions: Tuple of file extensions to process
            
        Returns:
            Dictionary containing results for all images
        """
        dir_path = Path(directory_path)
        if not dir_path.exists():
            return {
                'error': f'Directory not found: {directory_path}',
                'results': []
            }
        
        # Find all image files
        image_paths = []
        for ext in extensions:
            image_paths.extend(dir_path.glob(f'*{ext}'))
            image_paths.extend(dir_path.glob(f'*{ext.upper()}'))
        
        image_paths = [str(p) for p in image_paths]
        
        print(f"Found {len(image_paths)} images in {directory_path}")
        
        return self.classify_batch(image_paths, output_file)


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

