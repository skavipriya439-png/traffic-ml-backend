from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from ultralytics import YOLO
from sklearn.ensemble import RandomForestRegressor
import os

app = Flask(__name__)
CORS(app)

# Store latest detection result
latest_detection = None

# Load YOLO model
model = YOLO("yolo11n.pt")


# ================= HOME =================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "message": "Traffic ML Backend API is running"
    })


# ================= VIDEO UPLOAD =================

@app.route("/upload", methods=["POST"])
def upload_video():

    if "video" not in request.files:
        return jsonify({
            "status": "error",
            "message": "No video selected"
        }), 400

    video = request.files["video"]

    if video.filename == "":
        return jsonify({
            "status": "error",
            "message": "Empty video filename"
        }), 400

    video.save("uploaded_video.mp4")

    return jsonify({
        "status": "success",
        "message": "Video uploaded successfully"
    })


# ================= YOLO DETECTION =================

@app.route("/detect", methods=["GET", "POST"])
def detect_video():

    global latest_detection

    video_path = "uploaded_video.mp4"

    # If Flutter sends the video directly to /detect
    if "video" in request.files:

        video = request.files["video"]

        if video.filename == "":
            return jsonify({
                "status": "error",
                "message": "Empty video filename"
            }), 400

        video.save(video_path)

    # Check whether video exists
    if not os.path.exists(video_path):

        return jsonify({
            "status": "error",
            "message": "Please upload a video first"
        }), 400

    cars = []
    buses = []
    trucks = []
    motorcycles = []

    frame_count = 0

    try:

        results = model(
            video_path,
            stream=True,
            verbose=False
        )

        for result in results:

            frame_count += 1

            car_count = 0
            bus_count = 0
            truck_count = 0
            motorcycle_count = 0

            if result.boxes is not None:

                for box in result.boxes:

                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]

                    if class_name == "car":
                        car_count += 1

                    elif class_name == "bus":
                        bus_count += 1

                    elif class_name == "truck":
                        truck_count += 1

                    elif class_name == "motorcycle":
                        motorcycle_count += 1

            cars.append(car_count)
            buses.append(bus_count)
            trucks.append(truck_count)
            motorcycles.append(motorcycle_count)

            # Process first 30 frames
            if frame_count >= 30:
                break

    except Exception as error:

        return jsonify({
            "status": "error",
            "message": str(error)
        }), 500

    if frame_count == 0:

        return jsonify({
            "status": "error",
            "message": "Could not process video"
        }), 500

    # Calculate average vehicle count
    avg_cars = round(sum(cars) / len(cars))
    avg_buses = round(sum(buses) / len(buses))
    avg_trucks = round(sum(trucks) / len(trucks))
    avg_motorcycles = round(
        sum(motorcycles) / len(motorcycles)
    )

    total = (
        avg_cars
        + avg_buses
        + avg_trucks
        + avg_motorcycles
    )

    # Calculate traffic level
    if total <= 10:
        traffic_level = "Free"

    elif total <= 25:
        traffic_level = "Moderate"

    else:
        traffic_level = "Heavy"

    # Store latest result
    latest_detection = {
    "frame": frame_count,
    "cars": avg_cars,
    "buses": avg_buses,
    "trucks": avg_trucks,
    "motorcycles": avg_motorcycles,
    "total_vehicles": total,
    "traffic_level": traffic_level
}
    return jsonify({

        "status": "success",
        "message": "Video detection completed successfully",
        "cars": avg_cars,
        "buses": avg_buses,
        "trucks": avg_trucks,
        "motorcycles": avg_motorcycles,
        "total_vehicles": total,
        "traffic_level": traffic_level,
        "frames_processed": frame_count

    })


# ================= TRAFFIC DATA =================

@app.route("/traffic", methods=["GET"])
def traffic():

    # Use latest YOLO detection result
    if latest_detection is not None:

        return jsonify({

            "status": "success",
           "frame": latest_detection["frame"],
            "cars": latest_detection["cars"],
            "buses": latest_detection["buses"],
            "trucks": latest_detection["trucks"],
            "motorcycles": latest_detection["motorcycles"],
            "total_vehicles": latest_detection["total_vehicles"],
            "traffic_level": latest_detection["traffic_level"]

        })

    # Fallback to CSV data
    if not os.path.exists("traffic_data.csv"):

        return jsonify({
            "status": "error",
            "message": "Traffic data is not available"
        }), 404

    data = pd.read_csv("traffic_data.csv")
    latest = data.iloc[-1]

    return jsonify({

        "status": "success",
        "frame": int(latest["Frame"]),
        "cars": int(latest["Cars"]),
        "buses": int(latest["Buses"]),
        "trucks": int(latest["Trucks"]),
        "motorcycles": int(latest["Motorcycles"]),
        "total_vehicles": int(latest["Total_Vehicles"]),
        "traffic_level": latest["Traffic_Level"]

    })


# ================= CONGESTION PREDICTION =================

@app.route("/prediction", methods=["GET"])
def prediction():

    if not os.path.exists("traffic_data.csv"):

        return jsonify({
            "status": "error",
            "message": "Traffic data is not available"
        }), 404

    data = pd.read_csv("traffic_data.csv")

    traffic_values = data["Total_Vehicles"]

    if len(traffic_values) < 10:

        return jsonify({
            "status": "error",
            "message": "Not enough traffic data for prediction"
        }), 400

    # Current vehicle count
    if latest_detection is not None:
        current_vehicles = latest_detection["total_vehicles"]

    else:
        current_vehicles = int(traffic_values.iloc[-1])

    # Create previous traffic features
    data["Previous_1"] = traffic_values.shift(1)
    data["Previous_2"] = traffic_values.shift(2)
    data["Previous_3"] = traffic_values.shift(3)
    data["Previous_5"] = traffic_values.shift(5)
    data["Previous_10"] = traffic_values.shift(10)

    data = data.dropna()

    features = [
        "Previous_1",
        "Previous_2",
        "Previous_3",
        "Previous_5",
        "Previous_10"
    ]

    X = data[features]
    y = data["Total_Vehicles"]

    # Train Random Forest model
    prediction_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    prediction_model.fit(X, y)

    # Prepare prediction input
    input_data = pd.DataFrame([{

        "Previous_1": current_vehicles,
        "Previous_2": traffic_values.iloc[-2],
        "Previous_3": traffic_values.iloc[-3],
        "Previous_5": traffic_values.iloc[-5],
        "Previous_10": traffic_values.iloc[-10]

    }])

    next_prediction = prediction_model.predict(
        input_data
    )[0]

    # Predicted traffic level
    if next_prediction <= 10:
        predicted_level = "Free"

    elif next_prediction <= 25:
        predicted_level = "Moderate"

    else:
        predicted_level = "Heavy"

    return jsonify({

        "status": "success",
        "current_vehicles": int(current_vehicles),
        "predicted_vehicles": round(
            float(next_prediction), 2
        ),
        "predicted_traffic_level": predicted_level

    })


# ================= WHAT-IF SIMULATION =================

# ================= WHAT-IF SIMULATION =================

@app.route("/what if", methods=["GET"])
@app.route("/simulation", methods=["GET"])
def simulation():

    global latest_detection

    # Get current vehicle count
    if latest_detection is not None:

        current_vehicles = latest_detection["total_vehicles"]

    else:

        if not os.path.exists("traffic_data.csv"):

            return jsonify({
                "status": "error",
                "message": "Traffic data is not available"
            }), 404

        data = pd.read_csv("traffic_data.csv")

        current_vehicles = int(
            data["Total_Vehicles"].iloc[-1]
        )

    # Road blocked scenario
    increase_percent = 40

    simulated_vehicles = current_vehicles * (
        1 + increase_percent / 100
    )

    # Determine simulated traffic level
    if simulated_vehicles <= 10:

        simulated_level = "Free"

    elif simulated_vehicles <= 25:

        simulated_level = "Moderate"

    else:

        simulated_level = "Heavy"

    return jsonify({

        "status": "success",
        "scenario": "Road Blocked",
        "increase_percent": increase_percent,
        "current_vehicles": current_vehicles,
        "simulated_vehicles": round(
            simulated_vehicles, 2
        ),
        "simulated_traffic_level": simulated_level

    })


# ================= START SERVER =================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )