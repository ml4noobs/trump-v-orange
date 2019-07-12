# OK, time to deploy our model to the interwebs!
from flask import Flask, render_template, request
from fastai.vision import *
from pathlib import Path
import json
import os

# Model related stuff
CLASSES = ['orange', 'trump']
EXPORTED_LEARNER = Path("./package/")  # Path to export.pkl

# Our poor machine doesn't have a GPU, lol
defaults.device = torch.device("cpu")

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = "tmp/"


@app.route("/", methods=["GET"])
def root():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, TMP_DIR)

    # Create the dir
    if not os.path.exists(target):
        os.mkdir(target)

    # Upload the file, and process it
    for file in request.files.getlist("file"):
        filename = file.filename
        destination = "/".join([target, filename])
        file.save(destination)
        return json.dumps(predict_image(destination))


def predict_image(image_path):
    learn = load_learner(EXPORTED_LEARNER)
    _, _, losses = learn.predict(open_image(image_path))
    return {
        "predictions": sorted(
            zip(CLASSES, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        )
    }
