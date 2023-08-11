import numpy as np
from flask import Flask, request, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("ahi_model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")
@flask_app.route("/index")
def index():
    return render_template("index.html")

@flask_app.route("/landing")
def landing():
    return render_template("landing.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("landing.html", prediction_text="The AHI score is {}".format(prediction[0]))


if __name__ == "__main__":
    flask_app.run(debug=True)