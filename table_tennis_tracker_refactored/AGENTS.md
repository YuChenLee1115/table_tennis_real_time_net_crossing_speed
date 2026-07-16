# AGENTS.md

This file provides project guidance to Codex when working with code in this repository.

## Project Overview

This Python application reads webcam or video frames, detects and tracks a table-tennis ball, estimates speed, and writes statistics and charts. `main.py` is the command-line entry point and constructs `TableTennisTracker` from `src/core/tracker.py`.

### Implemented Capabilities

- **Ball Detection**: Frame differencing, background subtraction, contour scoring, and a Kalman filter
- **Speed Calculation**: Perspective correction, windowed estimates, and outlier filtering
- **Trajectory Management**: Prediction, interpolation, and continuity tracking
- **Concurrency**: A background frame-reader thread and a worker pool used by selected image operations

## Core Architecture

The system follows a modular architecture with clear separation of concerns:

### TableTennisTracker (src/core/tracker.py)
The main orchestrator that coordinates all components:
- Manages the main processing loop and state
- Integrates frame reading, detection, tracking, and visualization
- Handles session management and output generation
- Collects and manages speed data across net crossings

### Configuration System (src/core/config.py + config/default.json)
Dataclass-based configuration with JSON loading:
- `CameraConfig`: Camera/video input parameters
- `DetectionConfig`: Ball detection and FMO (Fast Moving Object) parameters
- `TrackingConfig`: Speed calculation and net crossing detection
- `VisualizationConfig`: Display colors and rendering options
- `IOConfig`: Data export and file handling

### Detection Pipeline
Two-stage ball detection:
1. **FMODetector** (src/detection/fmo_detector.py): Creates motion masks using background subtraction and morphological operations
2. **BallDetector** (src/detection/ball_detector.py): Extracts ball candidates from motion masks, scores them based on area/circularity/trajectory consistency

### Tracking & Speed Calculation
- **PerspectiveCorrector** (src/utils/perspective.py): Handles perspective transformation for accurate distance measurements
- **SpeedCalculator** (src/tracking/speed_calculator.py): Calculates ball speeds using perspective-corrected positions and timestamps
- **CrossingDetector** (src/tracking/crossing_detector.py): Detects when balls cross the net centerline

### Event System (src/core/events.py)
Dataclass-based event handling with structured data:
- `BallPosition`: Position data with timestamps
- `BallDetectionEvent`: Ball detection with metadata
- `NetCrossingEvent`: Net crossing events with speed data
- `EventManager`: Circular buffer for event history

### Data Export (src/io/data_exporter.py)
Asynchronous data export system:
- Generates matplotlib charts of speed data
- Exports CSV and TXT files with speed statistics
- Handles both real-time and video file output paths

## Common Commands

### Setup and Dependencies
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
# Real-time with webcam
python main.py

# Process video file
# Replace this placeholder with an existing readable video file.
python main.py --video path/to/video.mp4

# Debug mode with verbose output
python main.py --debug

# Custom camera/detection parameters
python main.py --fps 60 --width 1280 --height 720 --count 20 --timeout 0.2
```

### Standalone Diagnostics

`test_improvements.py` is an executable diagnostic script with its own `main()` function. It is not an automated test suite. Run it only after the dependencies from `requirements.txt` are installed in the active environment.

```bash
python test_improvements.py
```

The checked-in `tests/` directory is currently empty, so there is no runnable automated test suite there. Do not claim pytest coverage until test files are added.

### Code Quality (if tools are installed)
```bash
# Code style checking
flake8 src/ --max-line-length=100

# Type checking
mypy src/
```

## Key Implementation Details

### ROI (Region of Interest) System
The system uses configurable ROI ratios to focus detection on the table area:
- `roi_start_ratio`/`roi_end_ratio`: Horizontal boundaries (default: 40%-60% of frame width)
- `roi_bottom_ratio`: Vertical boundary (default: 85% of frame height)

### Speed Calculation Process
1. Ball positions are tracked in ROI coordinates
2. PerspectiveCorrector transforms positions to real-world coordinates using table dimensions
3. SpeedCalculator computes velocities between consecutive positions
4. Net crossing events trigger speed recording when balls cross the center zone

### Session Management
The tracker maintains session state:
- `is_counting_active`: Whether the system is collecting speed data
- `count_session_id`: Current session identifier
- `collected_net_speeds`/`collected_relative_times`: Speed data for current session
- `output_generated_for_session`: Prevents duplicate output generation

### Configuration Customization
- `config/default.json` exists, while `main.py` obtains a configuration object through `create_default_config()` and then applies command-line values.
- Use `python main.py --help` as the source of truth for supported command-line overrides.
- Do not claim a local configuration file is supported unless the loading code is added and verified.

## Generated Output Files

Successful runs can generate output in two modes; these paths are runtime output, not required source files:
- **Real-time mode**: Creates timestamped directories in `real_time_output/`
- **Video file mode**: Saves output alongside the source video file

Output includes:
- `*_chart.png`: Speed distribution histogram
- `*_data.csv`: Speed data in CSV format
- `*_data.txt`: Human-readable speed statistics

## Development Notes

### Performance Optimizations
- **OpenCV settings**: `PerformanceOptimizer` calls `cv2.setUseOptimized(True)` and attempts to set the OpenCV thread count with `cv2.setNumThreads(...)`.
- **Worker pool**: `PerformanceOptimizer` creates a `ThreadPoolExecutor` and uses it for selected chunked image operations.
- **Frame reader**: `FrameReader` reads frames on a background daemon thread and requests the AVFoundation capture API.
- **Memory and arrays**: The helper preallocates selected NumPy buffers and provides vectorized distance calculation.
- **Hardware boundary**: The source does not actively select a hardware acceleration backend. Do not infer hardware execution from platform-detection flags or comments.

### Detection Pipeline
- **Multi-Frame Analysis**: Uses multiple frame differences and background subtraction for motion detection
- **Adaptive Thresholding**: Dynamically adjusts detection sensitivity based on lighting conditions
- **Kalman Filtering**: Predictive filtering for ball position and velocity estimation
- **Multi-Criteria Scoring**: Ball candidates evaluated on area, circularity, solidity, trajectory consistency
- **Quality Assessment**: Real-time detection quality scoring based on multiple factors

### Trajectory Management
- **Trajectory Prediction**: Short-term trajectory prediction to handle temporary detection loss
- **Outlier Detection**: Statistical outlier filtering for trajectory points and speed calculations
- **Interpolation**: Interpolation for missing trajectory segments
- **Smoothing**: Multiple smoothing techniques including moving averages and trajectory fitting
- **Continuity Management**: Automatic trajectory recovery after interruptions

### Speed Calculation
- **Multi-Window Analysis**: Speed calculation using multiple trajectory window sizes
- **Perspective Integration**: Perspective correction using segmented path integration
- **Confidence Scoring**: Speed estimates include confidence ratings based on multiple factors
- **Adaptive Filtering**: Dynamic smoothing factors based on speed variability
- **Outlier Filtering**: Statistical outlier detection and filtering for speed values

### Debugging and Monitoring
- **Runtime Statistics**: Performance metrics exposed by components
- **Real-time Quality Metrics**: Live assessment of detection quality and system performance
- **Debug Mode**: Verbose logging with trajectory prediction and quality information
- **Performance Profiling**: Built-in performance analysis tools for optimization verification
