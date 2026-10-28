from datetime import date

from flask import Flask, escape, jsonify
from flask.json import JSONEncoder


class DateEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, date):
            return o.isoformat()
        return super().default(o)


app = Flask(__name__)
app.json_encoder = DateEncoder
STARTED = []


@app.before_first_request
def remember_start():
    STARTED.append(True)


@app.route("/hello/<name>")
def hello(name):
    return f"<h1>Hello {escape(name)}</h1>"


@app.route("/day")
def day():
    return jsonify({"day": date(2024, 5, 17)})
