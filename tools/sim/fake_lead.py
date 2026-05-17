#!/usr/bin/env python3
"""
Simula un coche parado (u en movimiento) delante del ego-vehicle escribiendo
en /tmp/fake_lead, que radard.py lee e inyecta en radarState.

No publica en ningún socket — evita el conflicto con radard.

Uso:
  python tools/sim/fake_lead.py --dist 50          # coche parado a 50 m
  python tools/sim/fake_lead.py --dist 30 --speed 5  # coche a 5 m/s
"""
import argparse
import os
import time
import cereal.messaging as messaging

_FAKE_LEAD_FILE = "/tmp/fake_lead"


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dist",  type=float, default=50.0,
                      help="Distancia inicial al obstáculo en metros (default: 50)")
  parser.add_argument("--speed", type=float, default=0.0,
                      help="Velocidad del obstáculo en m/s (default: 0 = parado)")
  args = parser.parse_args()

  sm = messaging.SubMaster(["carState"])

  dist = args.dist
  lead_speed = args.speed

  print(f"[FAKE_LEAD] Obstáculo a {dist:.1f} m, velocidad {lead_speed:.1f} m/s")
  print(f"[FAKE_LEAD] Escribiendo en {_FAKE_LEAD_FILE} — Ctrl+C para detener")

  last_t = time.monotonic()

  try:
    while True:
      sm.update(0)
      now = time.monotonic()
      dt = now - last_t
      last_t = now

      ego_speed = float(sm["carState"].vEgo) if sm.updated["carState"] else 0.0

      # Actualizar distancia relativa
      v_rel = lead_speed - ego_speed
      dist = max(0.0, dist + v_rel * dt)

      with open(_FAKE_LEAD_FILE, "w") as f:
        f.write(f"{dist:.3f} {lead_speed:.3f}\n")

      print(f"\r[FAKE_LEAD] ego={ego_speed*3.6:.1f} km/h | dRel={dist:.1f} m | vLead={lead_speed:.1f} m/s   ", end="")
      time.sleep(0.05)  # 20 Hz

  except KeyboardInterrupt:
    print("\n[FAKE_LEAD] Detenido — eliminando fichero de control")
    try:
      os.remove(_FAKE_LEAD_FILE)
    except FileNotFoundError:
      pass


if __name__ == "__main__":
  main()
