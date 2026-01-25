from flask import Flask,render_template

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

appliaction=Flask(__name__)

app=appliaction

##Route for home page
@app.route('/')
def index():
    return render_template('index.html')
