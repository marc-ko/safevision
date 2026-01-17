# -*- coding: utf-8 -*-
"""
Batch Content Classification Script
Process multiple images or directories and generate reports
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

import argparse
from pathlib import Path
from content_classifier import ThreePointRuleClassifier
import json
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description='Batch classify content based on Three Point Rule',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify single image
  python batch_classify.py image.jpg
  
  # Classify directory
  python batch_classify.py ./images --output results.json
  
  # Custom confidence threshold
  python batch_classify.py ./images --threshold 0.7
  
  # Show unsafe content only
  python batch_classify.py ./images --unsafe-only
  
  # Move safe images to safe folder
  python batch_classify.py ./images --safe-folder ./safe_images
  
  # Move both safe and unsafe images to separate folders
  python batch_classify.py ./images --safe-folder ./safe_images --unsafe-folder ./unsafe_images
        """
    )
    
    parser.add_argument('input', type=str, help='Image path or directory path')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output JSON results file (default: classification_results.json)')
    parser.add_argument('--threshold', '-t', type=float, default=None,
                       help='Confidence threshold (0.0-1.0, default: from CONFIDENCE_THRESHOLD env var or 0.5)')
    parser.add_argument('--unsafe-only', action='store_true',
                       help='Show unsafe content only')
    parser.add_argument('--safe-only', action='store_true',
                       help='Show safe content only')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed information')
    parser.add_argument('--safe-folder', '-s', type=str, default=None,
                       help='Folder to move safe images to (default: None, no moving)')
    parser.add_argument('--unsafe-folder', '-u', type=str, default=None,
                       help='Folder to move unsafe images to (default: None, no moving)')
    parser.add_argument('--genitalia-threshold', '-g', type=float, default=None,
                       help='Lower threshold for genitalia detection (default: from GENITALIA_THRESHOLD env var or confidence_threshold * 0.6)')
    parser.add_argument('--censor-folder', '-c', type=str, default=None,
                       help='Folder to save censored (blurred) versions of unsafe images (default: from CENSOR_FOLDER or None)')
    
    args = parser.parse_args()
    
    # Initialize classifier (will read from env vars if args are None)
    # Note: BLUR_INTENSITY and VIDEO_TRIM_DURATION are always read from module-level variables
    classifier = ThreePointRuleClassifier(
        confidence_threshold=args.threshold,
        safe_folder=args.safe_folder,
        unsafe_folder=args.unsafe_folder,
        genitalia_threshold=args.genitalia_threshold,
        censor_folder=args.censor_folder
    )
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Path not found: {args.input}")
        return
    
    # Determine output file (use fixed name for append mode)
    if args.output:
        output_file = args.output
    else:
        output_file = 'classification_results.json'  # Fixed filename for append mode
    
    # Process input
    if input_path.is_file():
        # Check if it's a video file
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv')
        is_video = input_path.suffix.lower() in video_extensions
        
        print(f"Processing: {args.input}")
        if is_video:
            result = classifier.classify_video(str(input_path))
        else:
            result = classifier.classify_image(str(input_path))
        
        # Filter results if needed
        if args.unsafe_only and result['safe']:
            return
        if args.safe_only and not result['safe']:
            return
        
        # Display result
        print_result(result, args.verbose)
        
        # Save result (append mode)
        classifier._save_results_append(output_file, result)
        print(f"\nResults saved to: {output_file}")
        
    elif input_path.is_dir():
        # Directory
        print(f"Processing directory: {args.input}")
        results = classifier.classify_directory(str(input_path), output_file=output_file)
        
        # Display summary
        print_summary(results, args.unsafe_only, args.safe_only, args.verbose)
        
    else:
        print(f"Error: {args.input} is not a valid file or directory")


def print_result(result: dict, verbose: bool = False):
    """Print single image/video classification result"""
    print(f"\n{'='*50}")
    
    # Determine if it's a video or image
    file_path = result.get('image_path') or result.get('video_path', 'N/A')
    is_video = 'video_path' in result
    
    if is_video:
        print(f"Video: {file_path}")
        if result.get('total_frames'):
            print(f"Total frames: {result.get('total_frames')}, FPS: {result.get('fps', 0):.2f}")
            print(f"Sampled frames: {result.get('sampled_frames', 0)}")
    else:
        print(f"Image: {file_path}")
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    if result['safe']:
        print("Safe (Three Point Rule: PASS)")
        if result.get('moved_to'):
            print(f"Moved to: {result['moved_to']}")
    else:
        print(f"Unsafe (Detected {result.get('total_points_detected', 0)} points)")
        if result.get('censored_to'):
            print(f"Censored version (full blur) saved to: {result['censored_to']}")
        if result.get('face_censored_to'):
            print(f"Face-censored version (face blur only) saved to: {result['face_censored_to']}")
        if result.get('moved_to'):
            print(f"Moved to: {result['moved_to']}")
        
        if result.get('detected_points'):
            print("\nDetected body parts:")
            for point in result['detected_points']:
                print(f"  • {point['name']} ({point['name_en']})")
                print(f"    Confidence: {point['confidence']:.2%}")
                print(f"    Points: {point['points']}")
                if point.get('frame') is not None:
                    print(f"    Frame: {point['frame']}")
    
    if verbose:
        if result.get('all_detections'):
            print(f"\nAll detections ({len(result['all_detections'])} items):")
            for det in result['all_detections']:
                print(f"  - {det['class']}: {det['confidence']:.2%}")
        if result.get('frame_results'):
            print(f"\nFrame-by-frame results:")
            for fr in result['frame_results']:
                print(f"  Frame {fr['frame']}: {fr['detected_points']} points detected")


def print_summary(results: dict, unsafe_only: bool, safe_only: bool, verbose: bool):
    """Print batch classification summary"""
    print(f"\n{'='*50}")
    print("Classification Summary")
    print(f"{'='*50}")
    total_files = results.get('total_images', 0) + results.get('total_videos', 0)
    print(f"Total images: {results.get('total_images', 0)}")
    if results.get('total_videos', 0) > 0:
        print(f"Total videos: {results.get('total_videos', 0)}")
    print(f"Safe (Three Point Rule: PASS): {results.get('safe_count', 0)}")
    print(f"Unsafe (Detected points): {results.get('unsafe_count', 0)}")
    print(f"Errors: {results.get('error_count', 0)}")
    
    if total_files > 0:
        safe_percent = (results.get('safe_count', 0) / total_files) * 100
        unsafe_percent = (results.get('unsafe_count', 0) / total_files) * 100
        print(f"\nSafe percentage: {safe_percent:.1f}%")
        print(f"Unsafe percentage: {unsafe_percent:.1f}%")
    
    # Show detailed results
    if verbose or unsafe_only or safe_only:
        print(f"\n{'='*50}")
        print("Detailed Results")
        print(f"{'='*50}")
        
        for result in results.get('results', []):
            if unsafe_only and result.get('safe', True):
                continue
            if safe_only and not result.get('safe', True):
                continue
            
            print_result(result, verbose=False)
            print()


if __name__ == '__main__':
    main()

