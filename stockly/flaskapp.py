from flask import Flask, render_template, request
from main import averagePercentChange, getAvgReturn, getAverageVol, returnPercent, is_real_stock

app = Flask(__name__)


@app.route("/")
def start():
    return render_template('homepage.html')


while True:
    app.run()
