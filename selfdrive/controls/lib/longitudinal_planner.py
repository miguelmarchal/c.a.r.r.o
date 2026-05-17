#!/usr/bin/env python3
import atexit
import json
import math
import numpy as np
import os
import threading
import time

import paho.mqtt.client as mqtt_lib

import cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, LongitudinalPlanSource
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog

A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5

# Lookup table for turns
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]

# ── Recomendación algoritmo externo + MLP velocidad segura ──────────────
_MLP_FALLBACK_KPH = 13.0  # fallback si el MLP no carga
_MLP_LOG_DIR      = os.path.dirname(os.path.abspath(__file__))
_MLP_PATH         = os.path.join(_MLP_LOG_DIR, 'mlp_danger_weights.npz')

# ── MQTT — recibe la señal 0/1 desde el servidor vía Mosquitto ───────────
# Broker público de pruebas Eclipse Mosquitto: cualquiera puede publicar,
# por eso el topic incluye un ID único para evitar colisiones con otros usuarios.
_MQTT_BROKER    = 'broker.hivemq.com'
_MQTT_PORT      = 1883
_MQTT_TOPIC     = 'openpilot/mmarc2026/ext_recommendation'   # ID único — no cambiar
_MQTT_KEEPALIVE = 60

_mqtt_lock  = threading.Lock()
_mqtt_value = 1   # valor por defecto: libre (safe)

def _on_mqtt_connect(client, userdata, connect_flags, reason_code, properties):
  if not reason_code.is_failure:
    client.subscribe(_MQTT_TOPIC)
    msg = f"[MQTT] Conectado a {_MQTT_BROKER}:{_MQTT_PORT} — suscrito a '{_MQTT_TOPIC}'"
    print(msg, flush=True)
    cloudlog.info(msg)
  else:
    msg = f"[MQTT] Error de conexion: {reason_code}"
    print(msg, flush=True)
    cloudlog.warning(msg)

def _on_mqtt_message(client, userdata, msg):
  global _mqtt_value
  try:
    val = int(msg.payload.decode().strip())
    if val in (0, 1):
      with _mqtt_lock:
        _mqtt_value = val
      txt = f"[MQTT] Mensaje recibido: ext_rec={val} ({'PELIGRO' if val == 0 else 'LIBRE'})"
      print(txt, flush=True)
      cloudlog.info(txt)
  except Exception as e:
    cloudlog.warning(f"[MQTT] Payload invalido '{msg.payload}': {e}")

def _on_mqtt_disconnect(client, userdata, disconnect_flags, reason_code, properties):
  txt = f"[MQTT] Desconectado ({reason_code}) — reconectando..."
  print(txt, flush=True)
  cloudlog.warning(txt)

def mqtt_start():
  """Llama esta funcion explicitamente al inicio del proceso (plannerd main)."""
  client = mqtt_lib.Client(mqtt_lib.CallbackAPIVersion.VERSION2)
  client.on_connect    = _on_mqtt_connect
  client.on_message    = _on_mqtt_message
  client.on_disconnect = _on_mqtt_disconnect
  client.reconnect_delay_set(min_delay=2, max_delay=30)

  def _run():
    print(f"[MQTT] Iniciando conexion a {_MQTT_BROKER}:{_MQTT_PORT} ...", flush=True)
    while True:
      try:
        client.connect(_MQTT_BROKER, _MQTT_PORT, _MQTT_KEEPALIVE)
        client.loop_forever()
      except Exception as e:
        txt = f"[MQTT] Conexion fallida: {e} — reintentando en 5 s"
        print(txt, flush=True)
        cloudlog.warning(txt)
        time.sleep(5)

  threading.Thread(target=_run, daemon=True, name="mqtt-planner").start()

def _read_ext_rec() -> int:
  with _mqtt_lock:
    return _mqtt_value
# ────────────────────────────────────────────────────────────────────────

def _load_mlp_weights():
  try:
    w = dict(np.load(_MLP_PATH))
    w['_n_layers'] = sum(1 for k in w if k.startswith('W'))
    cloudlog.info(f"[MLP] Cargado OK — {w['_n_layers']} capas — {_MLP_PATH}")
    return w
  except Exception as e:
    cloudlog.warning(f"[MLP] ERROR cargando {_MLP_PATH}: {e} — fallback {_MLP_FALLBACK_KPH} km/h")
    return None

def _mlp_safe_speed_kph(w, v_ego_ms: float, lead_dist_m: float,
                         lead_speed_ms: float, has_lead: bool,
                         a_ego_ms2: float = 0.0) -> float:
  """Inferencia numpy del MLP. Devuelve v_safe en km/h.
  Soporta modelos con 4 features (legacy) y 5 features (nuevo: incluye a_ego)."""
  if w is None:
    return _MLP_FALLBACK_KPH
  n_features = int(w.get('n_features', np.int32(4)))
  if n_features == 5:
    x = np.array([v_ego_ms, a_ego_ms2, lead_dist_m, lead_speed_ms, float(has_lead)], dtype=np.float32)
  else:
    x = np.array([v_ego_ms, lead_dist_m, lead_speed_ms, float(has_lead)], dtype=np.float32)
  x = (x - w['in_mean']) / (w['in_std'] + 1e-8)
  n = w['_n_layers']
  for i in range(n):
    x = x @ w[f'W{i}'] + w[f'b{i}']
    if i < n - 1:
      x = np.maximum(0.0, x)
  # El modelo recomienda v_safe — nunca acelerar en modo peligro
  return float(np.clip(x[0], 0.0, v_ego_ms * 3.6))
# ────────────────────────────────────────────────────────────────────────

def get_max_accel(v_ego):
  return np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)

def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3  # fitted from data using xx/projects/allow_throttle/compute_coast_accel.py

def limit_accel_in_turns(v_ego, angle_steers, a_target, CP):
  """
  This function returns a limited long acceleration allowed, depending on the existing lateral acceleration
  this should avoid accelerating when losing the target in turns
  """
  # FIXME: This function to calculate lateral accel is incorrect and should use the VehicleModel
  # The lookup table for turns should also be updated if we do this
  a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  a_y = v_ego ** 2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
  a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))

  return [a_target[0], min(a_target[1], a_x_allowed)]


class LongitudinalPlanner:
  def __init__(self, CP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.mpc = LongitudinalMpc(dt=dt)
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True

    self.a_desired = init_a
    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.prev_accel_clip = [ACCEL_MIN, ACCEL_MAX]
    self.output_a_target = 0.0
    self.output_should_stop = False

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)

    self._ext_rec_prev = 1      # estado anterior señal externa (1=libre, 0=peligro)
    self._mlp_w = _load_mlp_weights()
    self._mlp_log_counter = 0
    self._ext_rec_v_target = None     # velocidad objetivo (km/h) calculada por el MLP al inicio del evento
    self._ext_rec_v_cruise_ms = None  # v_cruise rate-limitado (m/s): baja desde v_ego hacia v_target
    self._mlp_records: list = []
    self._mlp_log_path = os.path.join(_MLP_LOG_DIR, f"mlp_log_{time.strftime('%Y%m%d_%H%M%S')}.json")
    atexit.register(self._write_mlp_log)

  @staticmethod
  def parse_model(model_msg):
    if (len(model_msg.position.x) == ModelConstants.IDX_N and
      len(model_msg.velocity.x) == ModelConstants.IDX_N and
      len(model_msg.acceleration.x) == ModelConstants.IDX_N):
      x = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.position.x)
      v = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.velocity.x)
      a = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.acceleration.x)
      j = np.zeros(len(T_IDXS_MPC))
    else:
      x = np.zeros(len(T_IDXS_MPC))
      v = np.zeros(len(T_IDXS_MPC))
      a = np.zeros(len(T_IDXS_MPC))
      j = np.zeros(len(T_IDXS_MPC))
    if len(model_msg.meta.disengagePredictions.gasPressProbs) > 1:
      throttle_prob = model_msg.meta.disengagePredictions.gasPressProbs[1]
    else:
      throttle_prob = 1.0
    return x, v, a, j, throttle_prob

  def update(self, sm):
    if len(sm['carControl'].orientationNED) == 3:
      accel_coast = get_coast_accel(sm['carControl'].orientationNED[1])
    else:
      accel_coast = ACCEL_MAX

    v_ego = sm['carState'].vEgo
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS

    # ── Recomendación algoritmo externo vía MQTT ─────────────────────────
    # Recibe 0/1 por MQTT (topic openpilot/ext_recommendation, broker Mosquitto).
    # 0 = peligro → limita velocidad crucero
    # 1 = libre   → openpilot actúa con normalidad
    # Transición 0→1: reset del filtro interno para recuperación inmediata
    # sin interferir con radar/AEB (que tienen prioridad total vía MPC).
    # ─────────────────────────────────────────────────────────────────────
    _ext_rec = _read_ext_rec()

    if _ext_rec == 0:
      lead = sm['radarState'].leadOne
      has_lead = bool(lead.status)
      v_ego_kph = v_ego * 3.6
      if has_lead:
        lead_dist_m   = float(lead.dRel)
        lead_speed_ms = float(lead.vLead)
      else:
        lead_dist_m   = 60.0
        lead_speed_ms = float(v_ego)

      if self._ext_rec_prev == 1 or self._ext_rec_v_target is None:
        # Primera vez en este evento: calcular target UNA SOLA VEZ y arrancar
        # el rate-limiter desde v_ego actual.
        self._ext_rec_v_target = _mlp_safe_speed_kph(
          self._mlp_w, float(v_ego), lead_dist_m, lead_speed_ms, has_lead,
          a_ego_ms2=float(sm['carState'].aEgo)
        )
        self._ext_rec_v_cruise_ms = float(v_ego)  # arranca desde velocidad actual
      elif has_lead:
        # Con líder real: actualizar target — la distancia cambia y es estable
        self._ext_rec_v_target = _mlp_safe_speed_kph(
          self._mlp_w, float(v_ego), lead_dist_m, lead_speed_ms, True,
          a_ego_ms2=float(sm['carState'].aEgo)
        )

      # Rate-limiter: bajar v_cruise a 1.0 m/s² hacia el target.
      # Esto hace que el obstáculo virtual del MPC baje GRADUALMENTE desde
      # safe_distance(v_ego) → el MPC frena desde el primer ciclo sin
      # el salto brusco que causaba "acercar → sobrepasar → parar".
      _EXT_DECEL_RATE = 1.0  # m/s²
      # El floor garantiza que el rate-limiter nunca baje de _MLP_FALLBACK_KPH
      # via la recomendación ext_rec. El MPC puede seguir frenando por debajo
      # (AEB, colisión real) porque sus restricciones internas son independientes
      # de v_cruise. Sin el floor, si el MLP devuelve 0 (v_ego≈0 → clip a 0),
      # el rate-limiter persiste en 0 y el coche no retoma velocidad al despejarse.
      v_target_ms = max(self._ext_rec_v_target, _MLP_FALLBACK_KPH) * CV.KPH_TO_MS
      self._ext_rec_v_cruise_ms = max(v_target_ms,
                                       self._ext_rec_v_cruise_ms - _EXT_DECEL_RATE * self.dt)
      v_cruise_kph = min(v_cruise_kph, self._ext_rec_v_cruise_ms * 3.6)
      v_cruise     = v_cruise_kph * CV.KPH_TO_MS
      self._mlp_records.append({
        "t":              round(time.monotonic(), 3),
        "v_ego_kph":      round(v_ego_kph, 2),
        "has_lead":       has_lead,
        "lead_dist_m":    round(lead_dist_m, 2),
        "lead_speed_kph": round(lead_speed_ms * 3.6, 2),
        "v_safe_kph":     round(self._ext_rec_v_target, 2),
        "v_cruise_rl_kph": round(self._ext_rec_v_cruise_ms * 3.6, 2),
        "v_cruise_kph":   round(v_cruise_kph, 2),
        "a_ego":          round(float(sm['carState'].aEgo), 3),
        "force_slow_decel": bool(sm['controlsState'].forceDecel),
        "allow_throttle": bool(self.allow_throttle),
      })

      # Log en la transición 1→0 y luego cada ~2 s (40 ciclos a 20 Hz)
      self._mlp_log_counter += 1
      if self._ext_rec_prev == 1 or self._mlp_log_counter >= 40:
        self._mlp_log_counter = 0
        cloudlog.info(
          f"[MLP] ext_rec=0 | MLP={'OK' if self._mlp_w else 'FALLBACK'} | "
          f"v_ego={v_ego_kph:.1f} kph | lead={'SI' if has_lead else 'NO'} "
          f"dRel={lead_dist_m:.1f}m | target={self._ext_rec_v_target:.1f} kph → v_cruise={v_cruise_kph:.1f} kph"
        )

    elif _ext_rec == 1 and self._ext_rec_prev == 0:
      # Transición 0→1: resetear estado interno al v_ego real actual
      # para que el MPC no arrastre la inercia de la velocidad reducida
      self.v_desired_filter.x = v_ego
      self.a_desired = max(self.a_desired, 0.0)
      self._ext_rec_v_target = None
      self._ext_rec_v_cruise_ms = None
      self._mlp_log_counter = 0
      self._write_mlp_log()   # escribir al terminar cada evento de peligro
      cloudlog.info(f"[MLP] ext_rec=1 (libre) | v_ego={v_ego*3.6:.1f} kph — reset filtro")

    self._ext_rec_prev = _ext_rec
    # ─────────────────────────────────────────────────────────────────────

    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off
    force_slow_decel = sm['controlsState'].forceDecel

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    # PCM cruise speed may be updated a few cycles later, check if initialized
    reset_state = reset_state or not v_cruise_initialized

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    accel_clip = [ACCEL_MIN, get_max_accel(v_ego)]
    steer_angle_without_offset = sm['carState'].steeringAngleDeg - sm['liveParameters'].angleOffsetDeg
    accel_clip = limit_accel_in_turns(v_ego, steer_angle_without_offset, accel_clip, self.CP)

    if reset_state:
      self.v_desired_filter.x = v_ego
      # Clip aEgo to cruise limits to prevent large accelerations when becoming active
      self.a_desired = np.clip(sm['carState'].aEgo, accel_clip[0], accel_clip[1])

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))
    _, _, _, _, throttle_prob = self.parse_model(sm['modelV2'])
    # Don't clip at low speeds since throttle_prob doesn't account for creep
    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    # En modo peligro externo: forzar allow_throttle=True para que el MPC
    # pueda reaccelerar hasta el target si v_ego baja demasiado.
    # Sin esto, throttle_prob bajo (rotonda) bloquea accel_clip[1] → el coche
    # solo puede frenar → cae por debajo del target → se detiene completamente.
    if self._ext_rec_v_target is not None:
      self.allow_throttle = True

    if not self.allow_throttle:
      clipped_accel_coast = max(accel_coast, accel_clip[0])
      clipped_accel_coast_interp = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED*2], [accel_clip[1], clipped_accel_coast])
      accel_clip[1] = min(accel_clip[1], clipped_accel_coast_interp)

    if force_slow_decel:
      if self._ext_rec_v_target is not None and not self.fcw:
        # En modo peligro externo SIN FCW: respetar nuestro target (rotonda sin obstáculo real).
        # force_slow_decel aquí es la curva/rotonda con throttle_prob bajo — no una emergencia.
        v_cruise = min(v_cruise, self._ext_rec_v_target * CV.KPH_TO_MS)
      else:
        # FCW activo O sin target externo: parada completa — comportamiento original de openpilot.
        v_cruise = 0.0

    self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)
    self.mpc.set_cur_state(self.v_desired_filter.x, self.a_desired)
    self.mpc.update(sm['radarState'], v_cruise, personality=sm['selfdriveState'].personality)

    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill

    # ── Reemplazo de trayectoria en modo peligro externo ─────────────────
    # El MPC planifica v→0 con v_cruise bajo (~11-13 km/h) porque el
    # obstáculo virtual safe_distance(v_target)≈14m queda "muy cerca".
    # Solución: sustituir la trayectoria por una rampa propia:
    #   · v_ego > target+0.1 m/s → frenar a -1.0 m/s² hasta target
    #   · v_ego < target-0.1 m/s → acelerar a +0.5 m/s² hasta target
    #   · en rango → mantener target con a=0
    # Esto garantiza coherencia (v y a consistentes) → sin alucinaciones.
    # Con obstáculo REAL (radar/FCW): el MPC manda, sin reemplazo.
    if self._ext_rec_v_target is not None:
      # Usar el mismo floor que el rate-limiter para no quedarse clavado a 0 cuando
      # v_ego → 0 y el MLP clipea v_safe a v_ego*3.6 ≈ 0.
      v_target_ms = max(self._ext_rec_v_target, _MLP_FALLBACK_KPH) * CV.KPH_TO_MS
      has_real_obstacle = (sm['radarState'].leadOne.status or
                           sm['radarState'].leadTwo.status or
                           self.fcw)
      if not has_real_obstacle:
        t_arr = np.array(CONTROL_N_T_IDX)
        if v_ego > v_target_ms + 0.1:
          # Frenar cómodamente hasta v_target
          _DECEL = 1.0
          v_traj = np.maximum(v_ego - _DECEL * t_arr, v_target_ms)
          a_traj = np.where(v_traj > v_target_ms + 0.01, -_DECEL, 0.0)
        elif v_ego < v_target_ms - 0.1:
          # Recuperar si v_ego cayó por debajo del objetivo
          _ACCEL = 0.5
          v_traj = np.minimum(v_ego + _ACCEL * t_arr, v_target_ms)
          a_traj = np.where(v_traj < v_target_ms - 0.01, _ACCEL, 0.0)
        else:
          # Mantener v_target
          v_traj = np.full(CONTROL_N, v_target_ms)
          a_traj = np.zeros(CONTROL_N)
        self.v_desired_trajectory = v_traj
        self.a_desired_trajectory = a_traj
        self.j_desired_trajectory = np.zeros(len(self.j_desired_trajectory))
    # ─────────────────────────────────────────────────────────────────────
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Interpolate 0.05 seconds and save as starting point for next iteration
    a_prev = self.a_desired
    self.a_desired = float(np.interp(self.dt, CONTROL_N_T_IDX, self.a_desired_trajectory))
    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.a_desired + a_prev) / 2.0

    action_t = self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target_mpc, output_should_stop_mpc = get_accel_from_plan(self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX,
                                                                        action_t=action_t, vEgoStopping=self.CP.vEgoStopping)
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = sm['modelV2'].action.shouldStop

    if sm['selfdriveState'].experimentalMode:
      output_a_target = min(output_a_target_e2e, output_a_target_mpc)
      self.output_should_stop = output_should_stop_e2e or output_should_stop_mpc
      if output_a_target < output_a_target_mpc:
        self.mpc.source = LongitudinalPlanSource.e2e
    else:
      output_a_target = output_a_target_mpc
      self.output_should_stop = output_should_stop_mpc

    for idx in range(2):
      accel_clip[idx] = np.clip(accel_clip[idx], self.prev_accel_clip[idx] - 0.05, self.prev_accel_clip[idx] + 0.05)
    self.output_a_target = np.clip(output_a_target, accel_clip[0], accel_clip[1])
    self.prev_accel_clip = accel_clip


  def _write_mlp_log(self):
    if not self._mlp_records:
      return
    try:
      with open(self._mlp_log_path, "w") as f:
        json.dump(self._mlp_records, f, indent=2)
      cloudlog.info(f"[MLP] Log → {self._mlp_log_path}  ({len(self._mlp_records)} registros)")
    except Exception as e:
      cloudlog.warning(f"[MLP] No se pudo guardar el log: {e}")

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState', 'selfdriveState', 'radarState'])

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.status
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)

    pm.send('longitudinalPlan', plan_send)