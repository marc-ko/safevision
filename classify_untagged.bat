@echo off
REM Batch script to classify all subfolders in untagged directory
REM This script scans all folders in untagged/ and processes them one by one
REM Each folder's results are organized with subfolder structure and zipped separately

set PYTHON_PATH=C:\Users\skip2\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe

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


