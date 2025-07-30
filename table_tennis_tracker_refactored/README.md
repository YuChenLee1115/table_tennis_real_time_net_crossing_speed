# Table Tennis Tracker - Refactored

## 專案結構

```
table_tennis_tracker_refactored/
├── src/                    # 源代碼
│   ├── core/              # 核心業務邏輯
│   ├── detection/         # 球體偵測相關
│   ├── tracking/          # 軌跡追蹤相關
│   ├── visualization/     # 視覺化與顯示
│   ├── io/               # 輸入輸出處理
│   └── utils/            # 工具函數
├── tests/                 # 測試文件
├── data/                  # 測試數據
├── config/               # 配置文件
├── main.py              # 主入口點
└── requirements.txt     # 依賴包列表
```

## 功能模組說明

### core/ - 核心模組
- `tracker.py`: 主要追蹤器類別
- `config.py`: 配置管理
- `events.py`: 事件記錄和處理

### detection/ - 偵測模組
- `ball_detector.py`: 球體偵測器
- `fmo_detector.py`: 快速移動物體偵測

### tracking/ - 追蹤模組
- `trajectory.py`: 軌跡管理
- `speed_calculator.py`: 速度計算
- `crossing_detector.py`: 中線穿越偵測

### visualization/ - 視覺化模組
- `renderer.py`: 視覺化渲染器
- `ui.py`: 用戶界面

### io/ - 輸入輸出模組
- `frame_reader.py`: 影像讀取器
- `data_exporter.py`: 數據輸出器

### utils/ - 工具模組
- `geometry.py`: 幾何計算
- `perspective.py`: 透視校正

## 使用方式

```bash
# 使用攝像頭
python main.py

# 使用影片檔案
python main.py --video path/to/video.mp4

# 開啟除錯模式
python main.py --debug

# 自定義參數
python main.py --fps 60 --width 1280 --height 720
```

## 安裝依賴

```bash
pip install -r requirements.txt
```