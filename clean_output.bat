@echo off
REM Clean all output folders and classification results
REM This script will delete all content inside output folders

echo ========================================
echo Cleaning Output Folders
echo ========================================
echo.

REM Clean censor_images folder
if exist "censor_images" (
    echo Cleaning censor_images...
    del /Q "censor_images\*" 2>nul
    if %errorlevel% == 0 (
        echo   [OK] censor_images cleaned
    ) else (
        echo   [INFO] censor_images is empty or already clean
    )
) else (
    echo   [INFO] censor_images folder does not exist
)

REM Clean face_censors folder
if exist "face_censors" (
    echo Cleaning face_censors...
    del /Q "face_censors\*" 2>nul
    if %errorlevel% == 0 (
        echo   [OK] face_censors cleaned
    ) else (
        echo   [INFO] face_censors is empty or already clean
    )
) else (
    echo   [INFO] face_censors folder does not exist
)

REM Clean safe_images folder
if exist "safe_images" (
    echo Cleaning safe_images...
    del /Q "safe_images\*" 2>nul
    if %errorlevel% == 0 (
        echo   [OK] safe_images cleaned
    ) else (
        echo   [INFO] safe_images is empty or already clean
    )
) else (
    echo   [INFO] safe_images folder does not exist
)

REM Clean unsafe_images folder
if exist "unsafe_images" (
    echo Cleaning unsafe_images...
    del /Q "unsafe_images\*" 2>nul
    if %errorlevel% == 0 (
        echo   [OK] unsafe_images cleaned
    ) else (
        echo   [INFO] unsafe_images is empty or already clean
    )
) else (
    echo   [INFO] unsafe_images folder does not exist
)

REM Clean untagged folder (if it's used for output)
if exist "untagged" (
    echo Cleaning untagged...
    del /Q "untagged\*" 2>nul
    if %errorlevel% == 0 (
        echo   [OK] untagged cleaned
    ) else (
        echo   [INFO] untagged is empty or already clean
    )
) else (
    echo   [INFO] untagged folder does not exist
)

REM Clean classification results JSON file
if exist "classification_results.json" (
    echo Cleaning classification_results.json...
    del /Q "classification_results.json" 2>nul
    if %errorlevel% == 0 (
        echo   [OK] classification_results.json deleted
    )
) else (
    echo   [INFO] classification_results.json does not exist
)

echo.
echo ========================================
echo Cleanup Complete!
echo ========================================
echo.
pause

