# PXL2LOC - Pixel to Location Electrode Coordinate Extractor

## Overview
PXL2LOC is a tool designed to extract and record the pixel coordinates of electrodes from EEG electrode net images. This is particularly useful for researchers and clinicians who need to map electrode positions from standardized electrode placement systems (like the GSN-256 or 10-20 systems) for EEG analysis and source localization.

This tool allows users to extract precise pixel coordinates from electrode net images and save them in a structured format for further analysis.

[Main Interface](docs/GUI.png)

## Project Structure
```
PXL2LOC/
├── auto_approach/          # Automated detection attempts (experimental)
│   ├── circle_detector.py
│   └── circle_detector_comprehensive.py
├── semi-manual_approach/   # Interactive GUI tool (recommended)
│   └── electrode_clicker_qt.py
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

1. Clone or download this repository
2. Create a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Recommended: Semi-Manual Approach (PyQt5 GUI)
The PyQt5 version provides the most user-friendly and feature-rich experience:

```bash
python3 semi-manual_approach/electrode_clicker_qt.py
```

#### Features:
- **File Selection**: Load any PNG or JPEG electrode net image
- **Two Insertion Modes**:
  - **Automatic**: Electrodes named E001-E256 sequentially
  - **Manual**: Custom electrode names (e.g., AF1, F7, Cz)
- **Interactive Controls**:
  - Left-click to place/select electrodes
  - Right-click to remove electrodes
  - Arrow keys to fine-tune positions
  - Adjustable circle size for different image scales
- **Auto-save**: Progress saved every 5 electrodes
- **Custom output filename**: Specify your own CSV filename

### Experimental: OpenCV Version
A simpler version using OpenCV directly:

```bash
python3 electrode_clicker.py
```

### Automated Approaches (Experimental)
The `auto_approach` directory contains experimental scripts for automated electrode detection:
- `circle_detector.py`: Basic Hough Circle detection
- `circle_detector_comprehensive.py`: Advanced multi-method detection

These are provided for reference but may not work reliably for all electrode net images.

## Output Format
The tool saves electrode coordinates in CSV format:

```csv
electrode_name,x,y
E001,123,456
E002,234,567
...
```

For manual mode with custom names:
```csv
electrode_name,x,y
Fp1,123,456
AF7,234,567
F7,345,678
...
```

## Tips for Best Results
1. Use high-resolution, clear images of electrode nets
2. Ensure good contrast between electrodes and background


## Requirements
- Python 3.9+
- See `requirements.txt` for complete list of dependencies

## Contributing
Feel free to submit issues or pull requests for improvements. Areas for contribution:
- Improved automated detection algorithms
- Support for additional electrode systems
- Export to other formats (e.g., MNI coordinates)
- 3D visualization of electrode positions

## License
This project is provided as-is for research and educational purposes. 
