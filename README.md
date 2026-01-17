# SafeVision - Three Point Rule Content Classification System

A tool to automatically detect whether content meets the "Three Point Rule" standard. Uses AI to detect specific body parts in images and determine if the content is safe.

#### Classify a Single Image
```bash
python batch_classify.py image.jpg
```

#### Classify an Entire Directory
```bash
python batch_classify.py ./images
```

#### Specify Output File
```bash
python batch_classify.py ./images --output my_results.json
```

#### Adjust Confidence Threshold
```bash
python batch_classify.py ./images --threshold 0.7
```

#### Show Only Unsafe Content
```bash
python batch_classify.py ./images --unsafe-only
```

#### Show Only Safe Content
```bash
python batch_classify.py ./images --safe-only
```

#### Show Detailed Information
```bash
python batch_classify.py ./images --verbose
```

#### Automatically Move Images to Target Folders
```bash
# Move safe images to the safe_images folder
python batch_classify.py ./images --safe-folder ./safe_images

# Move unsafe images to the unsafe_images folder
python batch_classify.py ./images --unsafe-folder ./unsafe_images

# Move both safe and unsafe images to separate folders
python batch_classify.py ./images --safe-folder ./safe_images --unsafe-folder ./unsafe_images

# Or use short options
python batch_classify.py ./images -s ./safe_images -u ./unsafe_images
```

### Method 2: Using content_classifier.py

#### As a Module
```python
from content_classifier import ThreePointRuleClassifier

# Initialize the classifier (optional: specify safe folder)
classifier = ThreePointRuleClassifier(
    confidence_threshold=0.5,
    safe_folder='./safe_images'  # Optional: automatically move safe images
)

# Classify a single image
result = classifier.classify_image('image.jpg')
print(f"Is safe: {result['safe']}")

# Batch classification
results = classifier.classify_batch(['img1.jpg', 'img2.jpg'])

# Classify a directory
results = classifier.classify_directory('./images', output_file='results.json')
```

#### Run Directly
```bash
# Single image
python content_classifier.py image.jpg

# Directory (batch processing)
python content_classifier.py ./images --batch
```

## Output Format

### Example of Single Image JSON Result

```json
{
  "safe": false,
  "image_path": "image.jpg",
  "total_points_detected": 2,
  "detected_points": [
    {
      "name": "Female Breasts",
      "name_en": "Female Breasts",
      "points": 2,
      "confidence": 0.85,
      "bbox": [100, 200, 300, 400]
    }
  ],
  "reason": "2 points detected, does not meet Three Point Rule",
  "moved_to": null
}
```

### Example of Batch Processing Results

```json
{
  "total_images": 10,
  "safe_count": 7,
  "unsafe_count": 2,
  "error_count": 1,
  "results": [
    {
      "safe": true,
      "image_path": "image1.jpg",
      "detected_points": [],
      "reason": "Three Point Rule passed, content is safe"
    },
    ...
  ]
}
```

## Parameter Descriptions

### confidence_threshold

- **Default**: 0.5
- **Range**: 0.0 - 1.0
- **How to set**:
  - Environment variable: `CONFIDENCE_THRESHOLD=0.5`
  - Command line: `--threshold 0.5` or `-t 0.5`
- **Description**: The detection confidence score must reach this threshold to be considered valid
  - Lower values (e.g., 0.3): More sensitive, may have more false positives
  - Higher values (e.g., 0.7): Stricter, fewer false positives but may miss detections

### safe_folder

- **Default**: None (do not move images)
- **Description**: Specify a folder path; safe images will be moved here automatically
  - The folder will be created if it doesn't exist
  - If a file with the same name exists at the destination, a timestamp is added to prevent overwriting
  - Example: `--safe-folder ./safe_images`

### unsafe_folder

- **Default**: None (do not move images)
- **How to set**:
  - Environment variable: `UNSAFE_FOLDER=unsafe_images`
  - Command line: `--unsafe-folder ./unsafe_images` or `-u ./unsafe_images`
- **Description**: Specify a folder path; unsafe images will be moved here automatically
  - The folder will be created if it doesn't exist
  - If a file with the same name exists at the destination, a timestamp is added to prevent overwriting
  - Example: `--unsafe-folder ./unsafe_images`

### genitalia_threshold

- **Default**: `CONFIDENCE_THRESHOLD * 0.6` (e.g., 0.3 if main threshold is 0.5)
- **Range**: 0.0 - 1.0
- **How to set**:
  - Environment variable: `GENITALIA_THRESHOLD=0.3`
  - Command line: `--genitalia-threshold 0.3` or `-g 0.3`
- **Description**: A lower threshold specifically for genitalia detection to catch edge cases
  - It's recommended to set this lower than `CONFIDENCE_THRESHOLD`
  - For example: if `CONFIDENCE_THRESHOLD=0.5`, set `GENITALIA_THRESHOLD=0.3` to catch lower-confidence genitalia detections

## What Are the "Three Points"?

The system detects the following body parts:

1. **EXPOSED_BREAST_F** - Exposed female breasts (2 points)
2. **EXPOSED_GENITALIA_F** - Exposed female genitalia (1 point)
3. **EXPOSED_GENITALIA_M** - Exposed male genitalia (1 point)

If any of these are detected, the content is marked as "unsafe".

## Notes

1. **First Use**: NudeNet will automatically download model files on first run, which may take some time
2. **Accuracy**: AI detection is not 100% accurate; manual review is recommended for borderline cases
3. **Privacy**: All processing is done locally, images are never uploaded to any server
4. **Performance**: Processing a large number of images may take some time; use batch mode for better efficiency

## Troubleshooting

### Installation Issues

If you encounter installation issues, make sure:
- Python version >= 3.7
- pip is installed
- Good network connection (needed for first model download)

### Detection Accuracy Problems

- Adjust the `confidence_threshold` parameter
- Check image quality and resolution
- Some artwork or unusual poses/angles may cause false positives or missed detections

## License

This project uses the open-source tool NudeNet; please comply with all related license terms.

## Contributions

Issues and Pull Requests are welcome!

