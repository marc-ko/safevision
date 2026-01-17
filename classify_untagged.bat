@echo off
REM Batch script to classify all subfolders in untagged directory
REM This script scans all folders in untagged/ and processes them one by one
REM Each folder's results are organized with subfolder structure and zipped separately

set PYTHON_PATH=python3

echo ========================================
echo Classifying Untagged Folders
echo ========================================
echo.
echo This will process all subfolders in the untagged/ directory
echo Each folder will be processed separately and zipped individually
echo.

"%PYTHON_PATH%" -c "from content_classifier import ThreePointRuleClassifier; classifier = ThreePointRuleClassifier(); classifier.classify_untagged_folders('untagged', 'classification_results.json')"

echo.
echo ========================================
echo Classification Complete!
echo ========================================
pause


