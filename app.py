from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
import json
import paho.mqtt.client as mqtt

app = Flask(__name__)

# Load the trained model
model = joblib.load('decision_tree_model.joblib')

# MQTT settings
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "symptoms"

# MQTT client
mqtt_client = mqtt.Client()

# Connect to the MQTT broker
mqtt_client.connect(MQTT_BROKER, MQTT_PORT)


# MQTT message callback
def on_message(client, userdata, message):
    payload = json.loads(message.payload)
    print("Received symptoms:", payload)


# Subscribe to MQTT topic
mqtt_client.on_message = on_message
mqtt_client.subscribe(MQTT_TOPIC)

# Load the dataset to get symptom fields
data = pd.read_csv("training.csv")
symptoms = data.columns[:-1]


# Home route
@app.route('/')
def home():
    return render_template('index.html', symptoms=symptoms)


# Prediction route
@app.route('/predict', methods=['POST'])

def predict():
    if request.method == 'POST':
        # Get the symptoms from the form
        selected_symptoms = request.form.getlist('symptoms[]')

        # Convert selected symptoms to binary representation
        symptoms_input = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]

        # Make prediction using the loaded model
        prediction = model.predict([symptoms_input])
        # Render the prediction result template with the prediction
        return render_template('result.html', prediction=prediction[0])




# Route to receive symptoms through MQTT
@app.route('/send-symptoms', methods=['POST'])
def send_symptoms():
    data = request.json
    mqtt_client.publish(MQTT_TOPIC, json.dumps(data))
    return "Symptoms sent through MQTT"


if __name__ == '__main__':
    app.run(debug=True)
