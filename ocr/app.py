import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flask import Flask, request, flash, jsonify
from werkzeug.utils import secure_filename
from ocr.externals.tesseract import Tesseract
from ocr.externals.detect_text import TextDetection
from configs import config
import cv2

text_detection = TextDetection(weights=config["text_detection_model"])
obj = Tesseract()

# The initiation of the flask app.
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config["max_image_size"]  # 16 MB Max Size.
app.debug = config["debug"]
app.secret_key = os.urandom(24)


# Check for allowed extensions
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config["allowed_extensions"]
    )


@app.route("/", methods=["POST"])
def upload_file():
    if "image" not in request.files:
        flash("No file part")
        return (
            jsonify({"output": "Invalid file.", "success": False}),
            422,
        )  # Message with status code.
    file = request.files["image"]

    if (file.filename).split(" ") != [file.filename]:
        flash("Rename Your file to be without spaces.")
        return (
            jsonify({"output": "No spaces in file name allowed.", "success": False}),
            422,
        )  # Message with status code.

    if file.filename == "":
        flash("No selected file")
        return (
            jsonify({"output": "Please select a file.", "success": False}),
            422,
        )  # Message with status code.

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(filename)
        results = test_prediction(file.filename)

        return jsonify({"output": results, "success": True}), 200
    else:
        return (
            jsonify(
                {
                    "output": "File extension is not allowed, use .jpg or .png",
                    "success": False,
                }
            ),
            422,
        )


def test_prediction(image):
    boxes = text_detection.detect(source=image, img_size=config["img_size"])
    return obj.inference(image=cv2.imread(image), boxes=boxes)


if __name__ == "__main__":
    app.run()
