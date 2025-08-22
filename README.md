# Microstructure AI

A machine learning project for analyzing and classifying microstructural images using computer vision and AI techniques.

## Overview

This project focuses on the analysis of microstructural images from the UHCSDB (Ultra-High Carbon Steel Database) using advanced image processing and machine learning algorithms. The goal is to develop automated methods for microstructure characterization and classification.

## Features

- **Image Processing**: Advanced image preprocessing using OpenCV and scikit-image
- **Feature Extraction**: GLCM (Gray-Level Co-occurrence Matrix) feature extraction
- **Machine Learning**: Classification models using scikit-learn
- **Data Visualization**: Interactive analysis with matplotlib and Jupyter notebooks
- **Large Dataset Support**: Handles thousands of micrograph images efficiently

## Dataset

The project uses the UHCSDB dataset containing:
- Over 600 micrograph images (.tif and .png formats)
- SQLite database with microstructure metadata
- Various steel microstructure types for analysis

## Requirements

- Python 3.9+
- OpenCV
- scikit-learn
- scikit-image
- numpy
- pandas
- matplotlib
- tqdm
- pillow

## Installation

```bash
pip install opencv-python scikit-learn scikit-image numpy pandas matplotlib tqdm pillow
```

## Usage

1. Clone the repository
2. Install dependencies
3. Open `microstructure_ai.ipynb` in Jupyter
4. Run the analysis cells

## Project Structure

```
microstructure-ai/
├── data/
│   └── UHCSDB/
│       ├── micrographs/     # Image files
│       └── microstructures.sqlite  # Database
├── microstructure_ai.ipynb  # Main analysis notebook
├── README.md
└── .gitignore
```

## Contributing

This is a research project focused on microstructure analysis. Contributions are welcome for:
- New feature extraction methods
- Improved classification algorithms
- Dataset expansion
- Documentation improvements

## License

This project is open source and available under the MIT License.
