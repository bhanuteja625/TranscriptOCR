
# Project-3-OCR-Team-1
# OCR Project: Automating Transcript Review Process

This project aims to automate the extraction of key information from transcripts submitted by applicants to streamline the review process for new applications. The project utilizes OCR (Optical Character Recognition) technology, specifically PaddleOCR fine-tuned on a custom transcripts dataset, along with additional algorithms for post-processing the OCR output.

## Features
- Extracts student names, universities, courses/subjects, grades, and GPAs from transcripts.
- Customizes PaddleOCR for improved accuracy on transcript data.
- Implements post-processing algorithms to refine the extracted information.

## Files and Directories

- `Fine_tuning_the_PaddleOCR_Model.ipynb`: Notebook containing code for fine-tuning the PaddleOCR model on a custom dataset.
- `OCR_capstone_project.ipynb`: Notebook containing the main project code, including post-processing blocks and end-to-end functionality.
- `README.md`: This file, containing an overview of the project and instructions for setup and usage.
- `app.py`: Python file for the Flask application, serving as the interface for users to upload and process images.
- `models/`: Directory containing the inference models for the detector and recognizer.
- `static/`: Directory containing static files for the web application, such as images and CSS/JavaScript files.
- `templates/`: Directory containing HTML templates for the web application.
- `test.jpg` and `test.png`: Sample images for testing the OCR model.
- `test_ocr.ipynb`: Notebook for testing the OCR model on the sample images.
- `utils.py`: Python file containing utility functions for the OCR project.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/bhanuteja625/TranscriptOCR.git
2. Install Dependencies:
   ```sh
   pip install -r requirements.txt

## Usage
- Prepare your custom transcripts dataset for fine-tuning PaddleOCR.
- Configure PaddleOCR for fine-tuning using the custom dataset.
- Train the model on the custom dataset to improve accuracy.
- Implement post-processing algorithms to refine the extracted information.
- Test the OCR model on sample transcripts to ensure accuracy.

## Contributors
- Bhanu Teja Nandina - Project Lead and Algorithm Development
- Indhiresh Reddy Lingareddy - Dataset Preparation
- Krishna Surendra Palepu - Post-processing
  
## Acknowledgements
- PaddleOCR - Open-source OCR toolkit
- Baidu Research - Developers of PaddleOCR
