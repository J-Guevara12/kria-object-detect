# integration_test.py

import os
import argparse
import time
import json
from datetime import datetime

# Importar las utilidades del archivo local
import connectivity_utils as cu

# --- Librerías específicas para el test ---
try:
    import numpy as np
    import cv2
except ImportError:
    np = None
    cv2 = None
    print("WARNING: numpy and cv2 not installed. Cannot create a temporary image file.")


def parse_args():
    """Define y parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Standalone test script for MQTT publishing and GCP file upload using connectivity_utils.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # MQTT Arguments
    parser.add_argument("--mqtt_host", type=str, default=cu.Config.MQTT_BROKER_HOST,
                        help="Host name or IP address of the MQTT broker.")
    parser.add_argument("--mqtt_port", type=int, default=cu.Config.MQTT_BROKER_PORT,
                        help="Port of the MQTT broker.")
    parser.add_argument("--mqtt_topic_base", type=str, default=cu.Config.MQTT_TOPIC_BASE,
                        help="MQTT base topic (e.g., kria/events).")
    
    # GCP Storage Arguments
    parser.add_argument("--gcp_bucket", type=str, required=True,
                        help="Name of the GCP bucket to upload the file to.")
    parser.add_argument("--gcp_blob", type=str, default="test_data/test_upload_manual.jpg",
                        help="Destination path/blob name in the GCP bucket.")
    
    # File Generation Argument
    parser.add_argument("--local_file", type=str, default="temp_test_capture.jpg",
                        help="Local filename for the temporary test image.")
    
    return parser.parse_args()

def create_test_file(filename="temp_test_capture.jpg"):
    """Crea una imagen de prueba simple."""
    if np is None or cv2 is None:
        print("Cannot create temporary image. Please ensure numpy and opencv-python are installed.")
        return None
        
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(img, "TEST", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(filename, img)
    return os.path.abspath(filename)

def main():
    args = parse_args()
    
    # 1. Aplicar la configuración de los argumentos a la clase Config
    cu.Config.MQTT_BROKER_HOST = args.mqtt_host
    cu.Config.MQTT_BROKER_PORT = args.mqtt_port
    cu.Config.MQTT_TOPIC_BASE = args.mqtt_topic_base
    cu.Config.GCP_BUCKET_NAME = args.gcp_bucket

    print("\n--- Starting Integration Test ---")
    print(f"GCP Bucket Target: {cu.Config.GCP_BUCKET_NAME}")
    print(f"MQTT Broker Target: {cu.Config.MQTT_BROKER_HOST}:{cu.Config.MQTT_BROKER_PORT}")
    
    local_file_path = None
    
    try:
        # ----------------------------------------------------
        # PHASE 1: File Preparation
        # ----------------------------------------------------
        local_file_path = create_test_file(args.local_file)
        if local_file_path:
            print(f"\n1. Test file created at: {local_file_path}")

        # ----------------------------------------------------
        # PHASE 2: Cloud Storage Upload Test
        # ----------------------------------------------------
        if local_file_path:
            print("\n2. Testing GCP Upload...")
            cu.upload_image_to_gcp(
                local_filepath=local_file_path,
                destination_blob_name=args.gcp_blob
            )
        else:
            print("\n2. GCP Upload skipped (No test file available).")

        # ----------------------------------------------------
        # PHASE 3: MQTT Publish Test
        # ----------------------------------------------------
        print("\n3. Testing MQTT Publish...")
        test_event = {
            "type": "test_event",
            "event": "manual_trigger",
            "object_class": "system_check",
            "timestamp": datetime.now().isoformat(),
            "message": "Integration test executed successfully from standalone script.",
            "status_code": 200
        }
        cu.publish_event_to_mqtt(test_event)
        
        # Give MQTT client time to process the loop
        time.sleep(1)

    finally:
        # ----------------------------------------------------
        # PHASE 4: Cleanup
        # ----------------------------------------------------
        print("\n4. Cleanup...")
        if local_file_path and os.path.exists(local_file_path):
            os.remove(local_file_path)
            print(f"   Removed temporary file: {args.local_file}")
            
        cu.cleanup_mqtt_client()
        
        print("\n--- Test Finished ---")

if __name__ == "__main__":
    main()
