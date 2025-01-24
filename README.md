# 行人偵測系統 - 階段一

本專案使用 YOLOv8 進行行人偵測。

## 環境設置

1. 安裝 Anaconda 或 Miniconda
2. 創建環境：
```bash
conda env create -f environment.yml
```
3. 啟動環境：
```bash
conda activate USR
```

## 使用說明

1. 將測試圖片或影片命名為 `test.jpg` 或 `test.mp4` 放在 `test_video` 資料夾
2. 執行測試程式：
```bash
python test_yolo.py
```

## 輸出說明

程式會輸出：
- 推論時間
- 檢測到的行人數量
- 結果圖片（result.jpg）
- 結果影片（result.mp4）