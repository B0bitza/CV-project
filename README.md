# Image Classifier App

This application allows users to upload an image and classify it as either "Muffin" or "Chihuahua". The backend is built using Flask and TensorFlow.

## Prerequisites

1. **Python Version**: Python 3.10
2. **Libraries**:
   - `tensorflowjs`: This will install the required dependencies.
   - `Pillow`: For image processing.

## Setup Instructions

### Step 1: Create  and activate the Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip3.10 install tensorflowjs Pillow
```

### Step 3: Generate the Comparator Model
```bash
python3.10 model.py
```

### Step 4: Run the Web Application
```bash
python3.10 webpage.py
```

### Usage
1. Upload an image using the form provided on the web interface.
2. The app will display a prediction based on the model.

### Additional Notes:
Ensure that you have all necessary files (`model.py`, `webui.py`, HTML templates, etc.) in place before running the application.
