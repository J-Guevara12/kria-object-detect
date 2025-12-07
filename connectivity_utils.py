import os
# ***********************************************************************
# External Integration Functions (MQTT & Cloud Storage)
# ***********************************************************************

import paho.mqtt.client as mqtt
from google.cloud import storage
import os
import json

# --- Parámetros de Configuración (Deben ser definidos en el entorno real) ---
# Usar una clase/estructura para mantener la configuración de manera limpia.

class Config:
    # MQTT
    MQTT_BROKER_HOST = os.environ.get("MQTT_HOST", "localhost")
    MQTT_BROKER_PORT = int(os.environ.get("MQTT_PORT", 1883))
    MQTT_TOPIC_BASE = os.environ.get("MQTT_TOPIC", "kria/events")
    MQTT_CLIENT = os.environ.get("MQTT_CLIENT", "usr")
    # GCP
    GCP_BUCKET_NAME = os.environ.get("GCP_BUCKET", "tu-nombre-de-bucket")
    
    # Cliente MQTT (Se inicializa una vez)
    mqtt_client = None

def get_mqtt_client():
    """Inicializa y conecta el cliente MQTT si aún no lo ha hecho."""
    if Config.mqtt_client is None and mqtt is not None:
        try:
            Config.mqtt_client = mqtt.Client(client_id="KriaDPUClient")
            # Opcional: Configurar credenciales si el broker lo requiere
            
            Config.mqtt_client.username_pw_set(Config.MQTT_CLIENT)
            Config.mqtt_client.connect(Config.MQTT_BROKER_HOST, Config.MQTT_BROKER_PORT, 60)
            Config.mqtt_client.loop_start() # Iniciar loop en segundo plano para manejar reconexiones
            print(f"MQTT Client connected to {Config.MQTT_BROKER_HOST}:{Config.MQTT_BROKER_PORT}")
        except Exception as e:
            print(f"Error connecting to MQTT Broker: {e}")
            Config.mqtt_client = None
    return Config.mqtt_client

def publish_event_to_mqtt(event_data: dict):
    """
    Función para publicar el evento JSON en un topic MQTT.
    """
    client = get_mqtt_client()
    if client:
        topic = f"kria-vision/{event_data['object_class']}/events/{event_data['event']}"
        payload = json.dumps(event_data)
        
        try:
            # QoS=1 (At least once) es un buen balance para eventos críticos
            client.publish(topic, payload, qos=1)
            # print(f"Published to MQTT Topic: {topic}") 
        except Exception as e:
            print(f"Error publishing MQTT message: {e}")

def upload_image_to_gcp(local_filepath: str, destination_blob_name: str):
    """
    Función para subir la imagen de la detección a Google Cloud Storage.
    """
    if storage is None:
        print("GCP Storage client not available. Skipping upload.")
        return
    
    try:
        # GCP asume que las credenciales (Application Default Credentials) están configuradas en el entorno
        storage_client = storage.Client()
        bucket = storage_client.bucket(Config.GCP_BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(local_filepath)
        print(f"Image uploaded to gs://{Config.GCP_BUCKET_NAME}/{destination_blob_name}")
        
        # Opcional: Generar la URL pública para el registro (si el bucket es público)
        # return blob.public_url

    except Exception as e:
        print(f"Error uploading image to GCP: {e}")

def cleanup_mqtt_client():
    """Detiene el loop y desconecta el cliente MQTT."""
    if Config.mqtt_client:
        Config.mqtt_client.loop_stop()
        Config.mqtt_client.disconnect()
        print("MQTT Client disconnected.")
