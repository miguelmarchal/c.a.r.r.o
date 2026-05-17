#!/usr/bin/env python3
"""
Envía señales 0/1 al longitudinal_planner de openpilot vía MQTT.

Uso:
  python3 mqtt_sender.py          # modo interactivo
  python3 mqtt_sender.py 0        # envía peligro y sale
  python3 mqtt_sender.py 1        # envía libre y sale
"""

import sys
import paho.mqtt.client as mqtt

BROKER = "broker.hivemq.com"
PORT   = 1883
TOPIC  = "openpilot/mmarc2026/ext_recommendation"


def send(client, value: int):
    result = client.publish(TOPIC, str(value), qos=1)
    result.wait_for_publish(timeout=5)
    label = "PELIGRO (frena)" if value == 0 else "LIBRE  (normal)"
    print(f"  → Enviado: {value}  [{label}]")


def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    print(f"Conectando a {BROKER}:{PORT} ...")
    client.connect(BROKER, PORT, keepalive=60)
    client.loop_start()
    print(f"Conectado. Topic: {TOPIC}\n")

    # Modo one-shot: python3 mqtt_sender.py 0  o  ...py 1
    if len(sys.argv) == 2 and sys.argv[1] in ("0", "1"):
        send(client, int(sys.argv[1]))
        client.loop_stop()
        client.disconnect()
        return

    # Modo interactivo
    print("Escribe  0  (peligro) o  1  (libre) y pulsa Enter.")
    print("Escribe  q  para salir.\n")
    try:
        while True:
            raw = input("señal> ").strip()
            if raw == "q":
                break
            if raw in ("0", "1"):
                send(client, int(raw))
            else:
                print("  Solo se acepta 0, 1 o q.")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        client.loop_stop()
        client.disconnect()
        print("\nDesconectado.")


if __name__ == "__main__":
    main()
