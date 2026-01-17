# Environment Variables Configuration

SafeVision 支援通過環境變數設置配置參數。所有環境變數使用大寫字母。

## 可用的環境變數

### CONFIDENCE_THRESHOLD
- **說明**: 一般身體部位檢測的置信度閾值
- **類型**: 浮點數 (0.0 - 1.0)
- **預設值**: 0.5
- **範例**: `CONFIDENCE_THRESHOLD=0.5`

### GENITALIA_THRESHOLD
- **說明**: 專門用於生殖器官檢測的較低閾值
- **類型**: 浮點數 (0.0 - 1.0)
- **預設值**: `CONFIDENCE_THRESHOLD * 0.6` (例如：0.3 如果 CONFIDENCE_THRESHOLD 是 0.5)
- **範例**: `GENITALIA_THRESHOLD=0.3`

### SAFE_FOLDER
- **說明**: 安全圖片移動到的資料夾路徑
- **類型**: 字串
- **預設值**: None（不移動）
- **範例**: `SAFE_FOLDER=safe_images`

### UNSAFE_FOLDER
- **說明**: 不安全圖片移動到的資料夾路徑
- **類型**: 字串
- **預設值**: None（不移動）
- **範例**: `UNSAFE_FOLDER=unsafe_images`

## 設置方法

### Windows PowerShell
```powershell
$env:CONFIDENCE_THRESHOLD="0.5"
$env:GENITALIA_THRESHOLD="0.3"
$env:SAFE_FOLDER="safe_images"
$env:UNSAFE_FOLDER="unsafe_images"
```

### Windows CMD
```cmd
set CONFIDENCE_THRESHOLD=0.5
set GENITALIA_THRESHOLD=0.3
set SAFE_FOLDER=safe_images
set UNSAFE_FOLDER=unsafe_images
```

### Linux/Mac
```bash
export CONFIDENCE_THRESHOLD=0.5
export GENITALIA_THRESHOLD=0.3
export SAFE_FOLDER=safe_images
export UNSAFE_FOLDER=unsafe_images
```

## 優先級

1. 命令行參數（最高優先級）
2. 環境變數
3. 預設值（最低優先級）

例如，如果同時設置了環境變數和命令行參數，命令行參數會覆蓋環境變數。

## 使用範例

```bash
# 設置環境變數後直接使用
$env:CONFIDENCE_THRESHOLD="0.4"
$env:GENITALIA_THRESHOLD="0.25"
python batch_classify.py ./images
```


