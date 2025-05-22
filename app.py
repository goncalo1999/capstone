import os
import json
import joblib
import pickle
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, TextField, FloatField, IntegerField,
    IntegrityError, CompositeKey
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect

# Database config
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')


class Prediction(Model):
    sku = TextField()
    time_key = IntegerField()
    pvp_is_competitorA = FloatField()
    pvp_is_competitorB = FloatField()
    pvp_is_competitorA_actual = FloatField(null=True)
    pvp_is_competitorB_actual = FloatField(null=True)

    class Meta:
        database = DB
        indexes = (
            (("sku", "time_key"), True),  # unique constraint
        )

DB.create_tables([Prediction], safe=True)

# Load models and data
model_A = joblib.load("model_compA.pkl")
model_B = joblib.load("model_compB.pkl")
prices = pd.read_csv("data/product_prices_leaflets.csv")

# Preprocess price data
# prices = prices[prices["discount"] >= 0].copy()
# prices["final_price"] = prices["pvp_was"] * (1 - prices["discount"])
prices["date"] = pd.to_datetime(prices["time_key"].astype(str), format="%Y%m%d")

# Create Flask app
app = Flask(__name__)

@app.after_request
def log_response(response):
    app.logger.warning(f"Response status: {response}")
    # app.logger.info(f"Response body: {response.get_data(as_text=True)}")
    return response

@app.route('/forecast_prices/', methods=['POST'])
def forecast_prices():
    try:
        payload = request.get_json()

        if not payload:
            log_response("Empty request body")
            return jsonify({"error": "Empty request body"}), 422
        
        sku_raw = payload.get('sku')
        time_key = int(payload['time_key'])

        if sku_raw is None or time_key is None:
            log_response("Missing required fields: 'sku' and 'time_key'")
            return jsonify({"error": "Missing required fields: 'sku' and 'time_key'"}), 422
        
        try:
            sku = int(sku_raw)
        except (TypeError, ValueError):
            log_response("Invalid SKU format")
            return jsonify({"error": "SKU must be a valid integer"}), 422
        
        try:
            target_date = pd.to_datetime(str(time_key), format="%Y%m%d") ## ver melhor isto
        except (TypeError, ValueError):
            log_response("Invalid time_key format")
            return jsonify({"error": "time_key in invalid format"}), 422
        
    except Exception:
        log_response("Error parsing JSON")
        return jsonify({"error": "Invalid input format"}), 422

    # Filter data for SKU
    df = prices[prices["sku"] == sku]
    df = df[df["competitor"].isin(["competitorA", "competitorB"])]
    if df.empty:
        log_response(f"SKU {sku} not found in the dataset.")
        return jsonify({"error": "SKU not found"}), 422

    df = df.pivot_table(index="date", columns="competitor", values="final_price").sort_index()
    if "competitorA" not in df.columns or "competitorB" not in df.columns:
        return jsonify({"error": "Missing one of the competitors data"}), 422

    df = df.rename(columns={"competitorA": "price_A", "competitorB": "price_B"})

    # Create lag and rolling features
    history = df[df.index < target_date].copy()

    history["final_price"] = history["price_A"]

    # Then build features using that column
    for lag in [1, 3, 7]:
        history[f"final_price_lag_{lag}"] = history["final_price"].shift(lag)
        history[f"final_price_roll_{lag}"] = history["final_price"].rolling(lag).mean()

    history = history.dropna()
    if history.empty:
        log_response("Not enough historical data")
        return jsonify({"error": "Not enough historical data"}), 422

    # Extract last row of features
    features = history.iloc[-1:][[col for col in history.columns if col.startswith("final_price_")]]


    # Predict using both models
    pred_A = float(model_A.predict(features)[0])
    pred_B = float(model_B.predict(features)[0])

    try:
        Prediction.create(
            sku=str(sku),
            time_key=time_key,
            pvp_is_competitorA=round(pred_A, 2),
            pvp_is_competitorB=round(pred_B, 2),
        )
    except IntegrityError:
        log_response(f"SKU {sku} and time_key {time_key} already exists in the database.")
        return jsonify({"error": "sku and time_key already exists"})
    
    response = jsonify({
        "sku": str(sku),
        "time_key": time_key,
        "pvp_is_competitorA": round(pred_A, 2),
        "pvp_is_competitorB": round(pred_B, 2)
    })

    
    return response


@app.route('/actual_prices/', methods=['POST'])
def actual_prices():
    try:
        payload = request.get_json()

        if not payload:
            log_response("Empty request body")
            return jsonify({"error": "Empty request body"}), 422

        required_fields = [
            "sku", "time_key",
            "pvp_is_competitorA_actual", "pvp_is_competitorB_actual"
        ]

        if not all(field in payload for field in required_fields):
            log_response("Missing one or more required fields")
            return jsonify({"error": "Missing one or more required fields"}), 422
        

        try:
            sku = str(payload["sku"])
            time_key = int(payload["time_key"])
            pvp_actual_A = float(payload["pvp_is_competitorA_actual"])
            pvp_actual_B = float(payload["pvp_is_competitorB_actual"])
        except (TypeError, ValueError):
            log_response("Incorrect field types")
            return jsonify({"error": "Incorrect field types"}), 422
        
    except (KeyError, ValueError, TypeError):
        log_response("Error parsing JSON")
        return jsonify({"error": "Invalid input format"}), 422

    try:
        record = Prediction.get(Prediction.sku == sku, Prediction.time_key == time_key)
        record.pvp_is_competitorA_actual = pvp_actual_A
        record.pvp_is_competitorB_actual = pvp_actual_B
        record.save()
    except Prediction.DoesNotExist:
        log_response(f"SKU {sku} and time_key {time_key} not found in the database.")
        return jsonify({"error": "SKU and time_key combination not found"}), 422
    
    response = jsonify({
        "sku": sku,
        "time_key": time_key,
        "pvp_is_competitorA": record.pvp_is_competitorA,
        "pvp_is_competitorB": record.pvp_is_competitorB,
        "pvp_is_competitorA_actual": pvp_actual_A,
        "pvp_is_competitorB_actual": pvp_actual_B
    })

    return response

@app.route('/list-db-contents')
def list_db_contents():
    return jsonify([model_to_dict(obs) for obs in Prediction.select()])


if __name__ == "__main__":
    # port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=5000, debug=False)


@app.cli.command("reset-db")
def reset_db():
    DB.drop_tables([Prediction])
    DB.create_tables([Prediction])
    print("Reset DB schema.")
