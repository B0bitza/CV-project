import os
from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

app = Flask(__name__)
uploader = 'images/'
app.config['uploader'] = uploader
os.makedirs(app.config['uploader'], exist_ok=True)
model = load_model("model/comparator.h5")


# Define a function to predict the class of an image
def check_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)[0]
    
    if len(prediction.shape) == 1:
        # For binary classification
        threshold = 0.5
        return "Muffin" if prediction > threshold else "Chihuahua"
    else:
        raise ValueError("Unexpected output shape from model.")

# Define the home page route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file and file.filename != "":
            file_path = os.path.join(app.config['uploader'], file.filename)
            file.save(file_path)
            
            result = check_image(file_path)
            return render_template("result.html", result=result, image_url=f"/images/{file.filename}")
    return render_template("index.html")

# Define a route to serve the uploaded images
@app.route('/images/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['uploader'], filename)

if __name__ == "__main__":
    app.run(debug=True)