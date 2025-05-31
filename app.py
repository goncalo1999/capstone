import os
import json
import joblib
import pickle
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, TextField, FloatField, IntegerField,
    IntegrityError
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
df_structures = pd.read_csv("product_structures_sales.csv")
df_structures["date"] = pd.to_datetime(df_structures["time_key"].astype(str), format="%Y%m%d")
df_structures = df_structures[df_structures["quantity"] >= 0]
print("df_structures columns:", df_structures.columns.tolist())

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
            target_date = pd.to_datetime(str(time_key), format="%Y%m%d")
        except (TypeError, ValueError):
            log_response("Invalid time_key format")
            return jsonify({"error": "time_key in invalid format"}), 422

    except Exception:
        log_response("Error parsing JSON")
        return jsonify({"error": "Invalid input format"}), 422

    preds = {}

    for comp_label, model in [("competitorA", model_A), ("competitorB", model_B)]:
        df_comp = prices[(prices["sku"] == sku) & (prices["competitor"] == comp_label)].copy()
        if df_comp.empty:
            log_response(f"SKU {sku} missing data for {comp_label}")
            return jsonify({"error": f"SKU missing data for {comp_label}"}), 422

        df_comp = df_comp.merge(
            df_structures[["sku", "date", "quantity", "structure_level_1", "structure_level_2"]],
            on=["sku", "date"],
            how="left"
        )

        df_comp = df_comp.sort_values(by="date")
        df_comp["quantity"] = df_comp["quantity"].fillna(0)
        df_comp["structure_level_1"] = df_comp["structure_level_1"].ffill().bfill()
        df_comp["structure_level_2"] = df_comp["structure_level_2"].ffill().bfill()

        df_comp = df_comp[df_comp["date"] < target_date]
        if df_comp.empty:
            log_response(f"Not enough historical data for {comp_label}")
            return jsonify({"error": f"Not enough historical data for {comp_label}"}), 422

        df_comp.set_index("date", inplace=True)
        df_comp["day_of_week"] = df_comp.index.dayofweek
        df_comp["month"] = df_comp.index.month

        for lag in [1, 7, 15]:
            df_comp[f"final_price_lag_{lag}"] = df_comp["final_price"].shift(lag)
            df_comp[f"final_price_roll_{lag}"] = df_comp["final_price"].rolling(lag).mean()
            df_comp[f"quantity_lag_{lag}"] = df_comp["quantity"].shift(lag)
            df_comp[f"quantity_roll_{lag}"] = df_comp["quantity"].rolling(lag).mean()

        df_comp.dropna(inplace=True)
        if df_comp.empty:
            log_response(f"Not enough data after creating features for {comp_label}")
            return jsonify({"error": f"Not enough data for {comp_label}"}), 422

        lag_features = [col for col in df_comp.columns if "lag" in col or "roll" in col]
        time_features = ["day_of_week", "month"]
        cat_features = ["structure_level_1", "structure_level_2"]

        df_comp[cat_features] = df_comp[cat_features].astype("category")
        features = df_comp.iloc[-1:][lag_features + time_features + cat_features]

        preds[comp_label] = float(model.predict(features)[0])

    try:
        Prediction.create(
            sku=str(sku),
            time_key=time_key,
            pvp_is_competitorA=round(preds["competitorA"], 2),
            pvp_is_competitorB=round(preds["competitorB"], 2),
        )
    except IntegrityError:
        log_response(f"SKU {sku} and time_key {time_key} already exists in the database.")
        return jsonify({"error": "sku and time_key already exists"})

    return jsonify({
        "sku": str(sku),
        "time_key": time_key,
        "pvp_is_competitorA": round(preds["competitorA"], 2),
        "pvp_is_competitorB": round(preds["competitorB"], 2)
    })



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
