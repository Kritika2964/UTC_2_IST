# RADAR Image Timestamp Conversion & Overlay

## Overview
This project automates the process of converting UTC timestamps in RADAR images to IST (Indian Standard Time) and overlays the converted timestamps back onto the images. The solution is designed to efficiently handle large sets of RADAR images, extracting timestamps using OCR, converting them, and then applying the converted timestamps to the images. The entire process is logged and saved in a pickle file for easy access and review.

## Table of Contents 
- Overview
- Importance
- How It Works
- Technologies Used

## Importance
- **Automation**: Reduces manual effort by automating timestamp extraction, conversion, and overlay.
- **Accuracy**: Ensures precise conversion of timestamps from UTC to IST.
- **Efficiency**: Capable of processing large datasets quickly and reliably.
- **Reproducibility**: The use of a pickle file for logging ensures that the results can be easily accessed and reviewed.

## How It Works
- **Image Preprocessing**: The images are processed using OpenCV to prepare them for OCR.
- OCR for Timestamp Extraction: Pytesseract is used to extract the UTC timestamps from the RADAR images.
- **Timestamp Conversion**: The extracted timestamps are converted from UTC to IST.
- **Overlaying Timestamps**: The converted IST timestamps are then overlaid onto the original RADAR images.
- **Logging**: The entire process, including the extracted text, converted timestamps, and status, is logged and stored in a pickle file.

## Technologies Used
- **Python**: Core programming language used for scripting and automation.
- **OpenCV**: Used for image processing and manipulation.
- **Pytesseract**: Used for Optical Character Recognition (OCR) to extract timestamps from images.
- **Pandas & NumPy**: Used for data manipulation and conversion processes.
- **Pickle**: Used for efficient logging and storage of the processed data.

## Prerequisites
- Python 3.x
- Tesseract OCR
- Required Python packages: pytesseract, numpy, opencv-python, imageio, dateutil, pandas, ftplib, pickle

## Installation

### Clone the repository:
git clone https://github.com/Kritika2964/UTC_2_IST.git

### Navigate to the project directory:
cd UTC_2_IST

### Install the required dependencies:
pip install -r requirements.txt

### Tesseract Setup:

- Ensure Tesseract OCR is installed on your system. You can download it from [https://github.com/tesseract-ocr/tesseract?tab=readme-ov-file](https://github.com/tesseract-ocr/tesseract?tab=readme-ov-file).

- Update the **pytesseract.pytesseract.tesseract_cmd** path in your code to point to your Tesseract installation directory.

## Usage
- Ensure you have your RADAR images and the necessary configuration files (config and station_product files) in the appropriate directories.
- To run the project, execute the following command:
    python scripts/utc_2_ist.py
- The processed images with IST timestamps will be saved in the designated output folder.
- The log file will be saved in the logs directory.
- The pickle file containing the log data will be saved in the logs directory.
- The original RADAR images will be preserved in the input directory.

## Project Structure

utc_2_ist/

├── Config/                     
├── Input/                      
├── Log/                        
│   ├── runtime/                
│   ├── date.pickle              
│   └── errorlog.csv            
├── scripts/                    
│   └── utc_2_ist.py            
├── requirements.txt            
└── README.md                   


## Configuration
- **Config File**: Customize your image processing settings by modifying the config file.
- **Station Product File**: Define the station and product-specific details in the station_product file.
- **Logging**: Logs will be stored in the log directory. Errors will be logged in errorlog.csv, and successful conversions will be logged in a pickle file named with the current date.

## Logging
- **Success Log**: Stored in Log/yyyy-mm-dd.pickle, this file contains detailed information about each processed file, including the filename, station ID, product ID, configuration ID, input image, extracted text, and status.
- **Error Log**: Any errors encountered during processing will be logged in Log/errorlog.csv.
- **Runtime Logs**: Cropped and preprocessed images are stored in the Log/runtime/ directory, along with the corresponding log CSV file.



