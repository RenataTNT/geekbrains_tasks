from flask import Flask, request, jsonify
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from process_data import process_input

# For logging
import logging
import traceback
from logging.handlers import RotatingFileHandler
from time import strftime, time

app = Flask(__name__)

xgb_ClaimInd_model = XGBClassifier()
booster = xgb.Booster()
booster.load_model('models/xgb_ClaimInd_model')
xgb_ClaimInd_model._Booster = booster

xgb_ClaimInd_model._le = LabelEncoder().fit([0,1,2,3,4,5,6,7,8,9,10])

# Logging
handler = RotatingFileHandler('app.log', maxBytes=100000, backupCount=5)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

@app.route("/")

def index():
    return "RenataTNT API"


@app.route("/predict", methods=['POST'])

def predict():
    json_input = request.get_json(force=True)

    # Request logging
    current_datatime = strftime('[%Y-%b-%d %H:%M:%S]')
    ip_address = request.headers.get("X-Forwarded-For", request.remote_addr)
    logger.info(f'{current_datatime} request from {ip_address}: {request.json}')
    start_prediction = time()

    id = json_input['ID']
    user_data = process_input(json_input)

    print(user_data)

    prediction_ClaimInd = xgb_ClaimInd_model.predict(user_data)

    ClaimInd = int(prediction_ClaimInd[0])
    print('prediction:', ClaimInd)

    result = {
        'ID': id,
        'ClaimInd': ClaimInd
    }


    # Response logging
    end_prediction = time()
    duration = round(end_prediction - start_prediction, 6)
    current_datatime = strftime('[%Y-%b-%d %H:%M:%S]')
    logger.info(f'{current_datatime} predicted for {duration} msec: {result}\n')

    return jsonify(result)

# @app.errorhandler(Exception)
# def exceptions(e):
#     current_datatime = strftime('[%Y-%b-%d %H:%M:%S]')
#     error_message = traceback.format_exc()
#     logger.error('%s %s %s %s %s 5xx INTERNAL SERVER ERROR\n%s',
#                  current_datatime,
#                  request.remote_addr,
#                  request.method,
#                  request.scheme,
#                  request.full_path,
#                  error_message)
#     return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
