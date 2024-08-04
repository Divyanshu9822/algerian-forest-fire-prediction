from flask import Flask, render_template, request
import pickle
from config import Config

app = Flask(__name__)
app.config.from_object(Config)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            temperature = float(request.form["temperature"])
            rh = float(request.form["rh"])
            ws = float(request.form["ws"])
            rain = float(request.form["rain"])
            ffmc = float(request.form["ffmc"])
            dmc = float(request.form["dmc"])
            isi = float(request.form["isi"])
            region = request.form["region"]

            if region == "Algeria":
                region = 0
            else:
                region = 1

            with open("models/ridge_cv.pkl", "rb") as model_file:
                model = pickle.load(model_file)
            with open("models/scaler.pkl", "rb") as scaler_file:
                scaler = pickle.load(scaler_file)

            input_data = scaler.transform(
                [[temperature, rh, ws, rain, ffmc, dmc, isi, region]]
            )

            prediction = model.predict(input_data)[0]

            print("Received input values:")
            print(f"Temperature: {temperature}")
            print(f"RH: {rh}")
            print(f"Ws: {ws}")
            print(f"Rain: {rain}")
            print(f"FFMC: {ffmc}")
            print(f"DMC: {dmc}")
            print(f"ISI: {isi}")
            print(f"Region: {region}")
            print(f"Prediction: {prediction}")

            return render_template("index.html", prediction=round(prediction, 2))

        except Exception as e:
            print(f"An error occurred: {e}")
            return render_template(
                "index.html", prediction="Error: Unable to make prediction."
            )
    return render_template("index.html", prediction=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8000)
