from flask import Flask, request, jsonify
import os
import pickle
import pandas as pd
import sqlite3


os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/", methods=['GET'])
def hello():
    return "Bienvenido a mi API del modelo advertising"

@app.route('/v2/predict', methods=['GET'])
def predict():
    try:
        model = pickle.load(open('data/advertising_model','rb'))

        tv = request.args.get('tv', None)
        radio = request.args.get('radio', None)
        newspaper = request.args.get('newspaper', None)

        if tv is None or radio is None or newspaper is None:
            return "Missing args, the input values are needed to predict"
        else:
            prediction = model.predict([[tv,radio,newspaper]])
            return "The prediction of sales investing that amount of money in TV, radio and newspaper is: " + str(round(prediction[0],2)) + 'k €'
    except Exception as err:
        return jsonify({"status": 500})

@app.route('/v2/ingest_data', methods=['POST'])
def post_ingest_data():
    try:
        connection = sqlite3.connect("./data/advertising.db")
        data = request.get_json()
        data_df = pd.DataFrame(data)
        data_df.to_sql("advertising", con=connection, if_exists="append")

    except Exception as err:
        return jsonify({"status": 500})

@app.route('/v2/retrain', methods=['PUT'])
def put_retrain():
    try:
        model = pickle.load(open('./data/advertising_model','rb'))
        connection = sqlite3.connect("./data/advertising.db")
        crsr = connection.cursor()
        crsr.execute("SELECT * FROM advertising")
        data = crsr.fetchall()
        # Obtenemos los nombres de las columnas de la tabla
        names = [description[0] for description in crsr.description]
        data_df =  pd.DataFrame(data,columns=names)
        X = data_df.drop("sales", axis=1)
        y = data_df["sales"]
        model.fit(X, y)
        with open('./data/advertising_model', "wb") as f:
            pickle.dump(data, f)

    except Exception as err:
        return jsonify({"status": 500})

app.run()

