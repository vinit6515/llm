from flask import Flask, redirect, request

app = Flask(__name__)

@app.route("/ask-csv", methods=["POST"])
def redirect_to_backend():
    return redirect("http://34.95.157.211:8000/ask-csv", code=307)
