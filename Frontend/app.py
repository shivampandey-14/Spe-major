from flask import Flask, render_template, request
import requests
import os

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        input_data = {
            "age": int(request.form["age"]),
            "balance": float(request.form["balance"]),
            "default": request.form["default"],
            "housing": request.form["housing"],
            "loan": request.form["loan"],
            "job": request.form["job"],
            "marital": request.form["marital"],
            "education": request.form["education"],
            "contact": request.form["contact"],
            "month": request.form["month"],
            "poutcome": request.form["poutcome"],
        }

        # Get the backend URL from environment variable or use default
        BACKEND_URL = os.getenv("BACKEND_URL", "http://172.17.0.2:5001")

        response = requests.post(
            f"{BACKEND_URL}/predict",
            json={"input": input_data, "model": request.form.get("model", "log_reg")},
        )

        if response.ok:
            result = response.json().get("prediction")

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
