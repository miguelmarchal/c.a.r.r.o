import math
import os
import time
import numpy as np

from collections import namedtuple
from panda3d.core import Vec3
from multiprocessing.connection import Connection

from metadrive.engine.core.engine_core import EngineCore
from metadrive.engine.core.image_buffer import ImageBuffer
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.obs.image_obs import ImageObservation

from openpilot.common.realtime import Ratekeeper

from openpilot.tools.sim.lib.common import vec3
from openpilot.tools.sim.lib.camerad import W, H

# Distancia en metros al obstáculo estático. 0 = sin obstáculo.
OBSTACLE_DISTANCE = float(os.environ.get("OBSTACLE_DISTANCE", "0"))
# 1 = activar NPC circulando en la rotonda (requiere traffic_density > 0 en el bridge)
ROUNDABOUT_NPC    = int(os.environ.get("ROUNDABOUT_NPC", "0"))
_FAKE_LEAD_FILE   = "/tmp/fake_lead"

C3_POSITION = Vec3(0.4, 0, 1.22)
C3_HPR = Vec3(0, 0,0)


def get_npc_conflict_lead(env, npc, max_dist_m: float = 60.0):
  """
  Calcula los valores que un radar + cámara reales reportarían para un NPC
  en la zona de conflicto de la rotonda.

  Valores físicamente correctos (igual que un radar real):
    d_rel    : distancia longitudinal (proyección delta sobre eje frontal del ego)
               > 0 = NPC delante, ≈ 0 = NPC lateral puro, < 0 = NPC detrás
    y_rel    : distancia lateral (convenio openpilot: + = derecha, - = izquierda)
    v_npc_fwd: componente longitudinal de la velocidad del NPC sobre el eje del ego
               (velocidad radial de cierre del NPC — lo que mide el radar Doppler)

  Solo inyecta si d_rel > 1 m (NPC genuinamente delante).
  Esto reemplaza el filtro de cono angular anterior, que daba falsos positivos
  cuando el NPC ya había pasado o estaba detrás después de entrar en la rotonda.
  """
  if npc is None:
    return None
  ego     = env.vehicle
  ego_pos = np.array(ego.position[:2], dtype=float)
  ego_fwd = np.array([math.cos(ego.heading_theta), math.sin(ego.heading_theta)])
  # vector perpendicular izquierda: rotar fwd 90° CCW
  ego_left = np.array([-ego_fwd[1], ego_fwd[0]])

  try:
    npc_pos = np.array(npc.position[:2], dtype=float)
    npc_vel = np.array(npc.velocity[:2], dtype=float)
  except Exception:
    return None

  delta = npc_pos - ego_pos
  dist  = float(np.linalg.norm(delta))
  if dist < 1.0 or dist > max_dist_m:
    return None

  # Componente longitudinal: positiva = NPC está delante del ego
  d_rel = float(np.dot(delta, ego_fwd))
  if d_rel < 1.0:
    return None  # NPC lateral puro o detrás — no hay conflicto longitudinal

  # Componente lateral (convenio openpilot: + = derecha, - = izquierda)
  y_rel = -float(np.dot(delta, ego_left))

  # Velocidad del NPC proyectada sobre el eje frontal del ego
  # (velocidad radial — lo que mide el radar Doppler real)
  v_npc_fwd = float(np.dot(npc_vel, ego_fwd))

  return (d_rel, y_rel, v_npc_fwd)


def spawn_static_obstacle(env, distance_m: float):
  """Spawna un coche estático (DefaultVehicle sin política) delante del ego-vehicle."""
  from metadrive.component.vehicle.vehicle_type import DefaultVehicle

  ego = env.agent
  heading = ego.heading_theta
  pos = ego.position + np.array([math.cos(heading), math.sin(heading)]) * distance_m

  obj = env.engine.spawn_object(
    DefaultVehicle,
    position=pos,
    heading=heading,
    vehicle_config=dict(render_vehicle=True),
  )
  print(f"[OBSTACLE] Coche estático spawneado a {distance_m:.0f}m — pos={pos}")
  return obj


class CircleActionController:
  """
  Hace circular un NPC aplicando acciones de dirección+aceleración en cada
  paso de física (before_step), en lugar de forzar posición directamente.
  El motor físico de MetaDrive aplica el movimiento — no lucha con Bullet.
  """
  SPEED_KMH  = 16.5   # velocidad objetivo en la rotonda
  STEER_MAG  = 0.15   # magnitud de dirección normalizada — radio ≈ 32m con wheelbase 3m

  def __init__(self, npc, clockwise: bool = False):
    self.npc   = npc
    # MetaDrive: +steer = giro a la izquierda, –steer = giro a la derecha
    self.steer = -self.STEER_MAG if clockwise else self.STEER_MAG

  def step(self):
    """Llamar justo ANTES de cada env.step() (cada 5 frames a 100 Hz)."""
    try:
      vel = self.npc.velocity
      speed_ms = float(math.sqrt(vel[0]**2 + vel[1]**2))
    except Exception:
      speed_ms = 0.0
    target_ms = self.SPEED_KMH / 3.6
    if speed_ms < target_ms - 0.5:
      throttle = 0.5
    elif speed_ms > target_ms + 0.5:
      throttle = -0.3
    else:
      throttle = 0.0
    try:
      self.npc.before_step([self.steer, throttle])
    except Exception:
      pass


def spawn_roundabout_npc(env):
  """
  Spawna exactamente 1 NPC en el anillo de la rotonda.
  Devuelve (npc, CircleActionController).
  El controlador usa before_step([steer, throttle]) en cada paso de física.
  """
  from metadrive.component.vehicle.vehicle_type import DefaultVehicle

  rn = env.engine.current_map.road_network
  target_lane = None

  # Paso 1: lanes donde AMBOS nodos contienen 'O' → anillo puro
  for start_node, end_dict in rn.graph.items():
    if 'O' not in start_node:
      continue
    for end_node, lanes in end_dict.items():
      if 'O' in end_node and lanes:
        target_lane = lanes[0]
        break
    if target_lane is not None:
      break

  # Paso 2: fallback — cualquier lane del bloque 'O'
  if target_lane is None:
    for start_node, end_dict in rn.graph.items():
      if 'O' not in start_node:
        continue
      for end_node, lanes in end_dict.items():
        if lanes:
          target_lane = lanes[0]
          break
      if target_lane is not None:
        break

  if target_lane is None:
    print("[NPC] WARN: No se encontró lane de rotonda — NPC no spawneado.")
    return None, None

  try:
    t_mid   = target_lane.length * 0.3
    pos     = target_lane.position(t_mid, 0)
    heading = float(target_lane.heading_theta_at(t_mid))
    # Dos puntos para determinar sentido de giro (CW vs CCW)
    h_start = float(target_lane.heading_theta_at(0.0))
    h_end   = float(target_lane.heading_theta_at(target_lane.length * 0.99))
  except Exception as e:
    print(f"[NPC] Error obteniendo posición en lane: {e}")
    return None, None

  # Si el heading gira a la derecha a lo largo de la lane → CW
  delta_h   = (h_end - h_start + math.pi) % (2 * math.pi) - math.pi
  clockwise = delta_h < 0

  npc = env.engine.spawn_object(
    DefaultVehicle,
    position=pos,
    heading=heading,
    vehicle_config=dict(render_vehicle=True, enable_reverse=False),
  )

  ctrl = CircleActionController(npc, clockwise=clockwise)
  print(f"[NPC] Spawneado en rotonda — pos=({pos[0]:.1f},{pos[1]:.1f}) "
        f"heading={math.degrees(heading):.1f}° sentido={'CW' if clockwise else 'CCW'}")
  return npc, ctrl


metadrive_simulation_state = namedtuple("metadrive_simulation_state", ["running", "done", "done_info"])
metadrive_vehicle_state = namedtuple("metadrive_vehicle_state", ["velocity", "position", "bearing", "steering_angle", "nav_arrow"])
def apply_metadrive_patches(arrive_dest_done=True):
  # By default, metadrive won't try to use cuda images unless it's used as a sensor for vehicles, so patch that in
  def add_image_sensor_patched(self, name: str, cls, args):
    if self.global_config["image_on_cuda"]:# and name == self.global_config["vehicle_config"]["image_source"]:
        sensor = cls(*args, self, cuda=True)
    else:
        sensor = cls(*args, self, cuda=False)
    assert isinstance(sensor, ImageBuffer), "This API is for adding image sensor"
    self.sensors[name] = sensor

  EngineCore.add_image_sensor = add_image_sensor_patched

  # we aren't going to use the built-in observation stack, so disable it to save time
  def observe_patched(self, *args, **kwargs):
    return self.state

  ImageObservation.observe = observe_patched

  # disable destination, we want to loop forever
  def arrive_destination_patch(self, *args, **kwargs):
    return False

  if not arrive_dest_done:
    MetaDriveEnv._is_arrive_destination = arrive_destination_patch

  # MetaDrive 0.4.2.3: los NPCs de tráfico se crean sin render_vehicle en su
  # vehicle_config → KeyError en base_vehicle.__init__. Parcheamos la clase
  # para que use False como valor por defecto — los assets mínimos no incluyen
  # el modelo ferra, así que render_vehicle=True crashea al cargar la rueda.
  from metadrive.component.vehicle.base_vehicle import BaseVehicle
  _orig_bv_init = BaseVehicle.__init__

  def _patched_bv_init(self, vehicle_config=None, **kwargs):
    if vehicle_config is not None and "render_vehicle" not in vehicle_config:
      try:
        vehicle_config["render_vehicle"] = False
      except Exception:
        pass
    _orig_bv_init(self, vehicle_config=vehicle_config, **kwargs)

  BaseVehicle.__init__ = _patched_bv_init

def metadrive_process(dual_camera: bool, config: dict, camera_array, wide_camera_array, image_lock,
                      controls_recv: Connection, simulation_state_send: Connection, vehicle_state_send: Connection,
                      exit_event, op_engaged, test_duration, test_run):
  arrive_dest_done = config.pop("arrive_dest_done", True)
  apply_metadrive_patches(arrive_dest_done)

  road_image = np.frombuffer(camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))
  if dual_camera:
    assert wide_camera_array is not None
    wide_road_image = np.frombuffer(wide_camera_array.get_obj(), dtype=np.uint8).reshape((H, W, 3))

  env = MetaDriveEnv(config)

  def get_current_lane_info(vehicle):
    _, lane_info, on_lane = vehicle.navigation._get_current_lane(vehicle)
    lane_idx = lane_info[2] if lane_info is not None else None
    return lane_idx, on_lane

  obstacle        = None
  roundabout_npc  = None
  npc_ctrl        = None

  # Seguimiento del NPC de rotonda: inyectamos fake_lead solo cuando el NPC
  # está físicamente en el carril del ego (|y_rel| < PATH_Y_THRESHOLD).
  # Una vez que el NPC cruza, marcamos _npc_cleared=True y mantenemos CLEAR
  # mientras el NPC siga dentro del FOV frontal de la cámara (~40° semi-ángulo).
  # En cuanto sale del FOV, esperamos NPC_MODEL_FORGET_S para que el modelo
  # olvide el track temporal, y entonces quitamos el CLEAR → ACC reanuda.
  PATH_Y_THRESHOLD    = 3.5   # metros — ancho de carril típico
  NPC_CAM_HALF_FOV    = math.radians(40)  # semi-ángulo de la cámara frontal
  NPC_MODEL_FORGET_S  = 2.5   # segundos que el modelo tarda en olvidar el NPC
  _npc_was_in_path    = False  # el NPC estuvo alguna vez en el carril del ego
  _npc_cleared        = False  # el NPC ya cruzó y despejó el carril
  _npc_fov_exit_time  = None   # monotonic: cuándo salió el NPC del FOV

  def reset():
    nonlocal obstacle, roundabout_npc, npc_ctrl
    nonlocal _npc_was_in_path, _npc_cleared, _npc_fov_exit_time
    env.reset()
    env.vehicle.config["max_speed_km_h"] = 25
    lane_idx_prev, _ = get_current_lane_info(env.vehicle)

    obstacle            = None
    roundabout_npc      = None
    npc_ctrl            = None
    _npc_was_in_path    = False
    _npc_cleared        = False
    _npc_fov_exit_time  = None

    if OBSTACLE_DISTANCE > 0:
      obstacle = spawn_static_obstacle(env, OBSTACLE_DISTANCE)

    if ROUNDABOUT_NPC:
      roundabout_npc, npc_ctrl = spawn_roundabout_npc(env)

    simulation_state = metadrive_simulation_state(
      running=True,
      done=False,
      done_info=None,
    )
    simulation_state_send.send(simulation_state)

    return lane_idx_prev

  lane_idx_prev = reset()
  start_time = None

  def get_cam_as_rgb(cam):
    cam = env.engine.sensors[cam]
    cam.get_cam().reparentTo(env.vehicle.origin)
    cam.get_cam().setPos(C3_POSITION)
    cam.get_cam().setHpr(C3_HPR)
    img = cam.perceive(to_float=False)
    if not isinstance(img, np.ndarray):
      img = img.get() # convert cupy array to numpy
    return img

  rk = Ratekeeper(100, None)

  steer_ratio = 12
  vc = [0,0]

  try:
    while not exit_event.is_set():
      vehicle_state = metadrive_vehicle_state(
        velocity=vec3(x=float(env.vehicle.velocity[0]), y=float(env.vehicle.velocity[1]), z=0),
        position=env.vehicle.position,
        bearing=float(math.degrees(env.vehicle.heading_theta)),
        steering_angle=env.vehicle.steering * env.vehicle.MAX_STEERING,
        nav_arrow=list(env.vehicle.navigation.navi_arrow_dir)
      )
      vehicle_state_send.send(vehicle_state)

      if controls_recv.poll(0):
        while controls_recv.poll(0):
          steer_angle, gas, should_reset = controls_recv.recv()

        steer_metadrive = steer_angle * 1 / (env.vehicle.MAX_STEERING * steer_ratio)
        steer_metadrive = np.clip(steer_metadrive, -1, 1)

        vc = [steer_metadrive, gas]

        if should_reset:
          lane_idx_prev = reset()
          start_time = None

      is_engaged = op_engaged.is_set()
      if is_engaged and start_time is None:
        start_time = time.monotonic()

      if rk.frame % 5 == 0:
        # Aplicar acción al NPC ANTES del paso de física
        if npc_ctrl is not None:
          npc_ctrl.step()
        _, _, terminated, _, _ = env.step(vc)
        timeout = True if start_time is not None and time.monotonic() - start_time >= test_duration else False
        lane_idx_curr, on_lane = get_current_lane_info(env.vehicle)
        out_of_lane = lane_idx_curr != lane_idx_prev or not on_lane
        lane_idx_prev = lane_idx_curr

        if terminated or ((out_of_lane or timeout) and test_run):
          if terminated:
            done_result = env.done_function("default_agent")
          elif out_of_lane:
            done_result = (True, {"out_of_lane" : True})
          elif timeout:
            done_result = (True, {"timeout" : True})

          simulation_state = metadrive_simulation_state(
            running=False,
            done=done_result[0],
            done_info=done_result[1],
          )
          simulation_state_send.send(simulation_state)

        if dual_camera:
          wide_road_image[...] = get_cam_as_rgb("rgb_wide")
        road_image[...] = get_cam_as_rgb("rgb_road")
        image_lock.release()

      # ── Fake-lead + ext_recommendation: obstáculo estático + NPC de rotonda ──
      if obstacle is not None:
        ego_pos = env.vehicle.position
        obs_pos = obstacle.position
        dist = math.sqrt((ego_pos[0] - obs_pos[0])**2 + (ego_pos[1] - obs_pos[1])**2)
        with open(_FAKE_LEAD_FILE, "w") as f:
          f.write(f"{dist:.3f} 0.0\n")
        # No escribir ext_recommendation: el MPC de openpilot frena solo con el
        # fake_lead. El sistema MLP/ext_rec es para el NPC de rotonda, no para
        # obstáculos estáticos. Así el archivo puede controlarse externamente.
      elif roundabout_npc is not None:
        result = get_npc_conflict_lead(env, roundabout_npc)
        inject = False

        if not _npc_cleared:
          if result is not None:
            d_rel, y_rel, v_npc_fwd = result
            inject = True

            in_path = abs(y_rel) < PATH_Y_THRESHOLD
            if in_path and not _npc_was_in_path:
              ego_spd = float(np.linalg.norm(env.vehicle.velocity[:2]))
              print(f"[NPC] Cruzando carril del ego — dRel={d_rel:.1f}m yRel={y_rel:.1f}m "
                    f"vNPC={v_npc_fwd*3.6:.1f}km/h ego={ego_spd*3.6:.1f}km/h")
            if in_path:
              _npc_was_in_path = True
              # NPC en el carril del ego: usar d_rel real para frenado preciso
              d_rel_inject = d_rel
            elif _npc_was_in_path:
              # el NPC estuvo en el carril y ahora se alejó lateralmente → despejó
              _npc_cleared = True
              inject = False
              print(f"[NPC] Despejó el carril — dRel={d_rel:.1f}m yRel={y_rel:.1f}m — CLEAR por FOV+modelo")
            else:
              # NPC aún lateral: usar distancia 3D total como d_rel.
              # Esto activa el MLP con tiempo suficiente para reducir la velocidad
              # gradualmente desde lejos, en vez de detectar el NPC a 3-5m de la
              # entrada de la rotonda y frenar en seco.
              ego_xy = np.array(env.vehicle.position[:2], dtype=float)
              npc_xy = np.array(roundabout_npc.position[:2], dtype=float)
              d_rel_inject = float(min(np.linalg.norm(npc_xy - ego_xy), 55.0))
          elif _npc_was_in_path:
            # result=None: NPC detrás/lejos, pero antes cruzó el carril → despejó
            _npc_cleared = True
            print(f"[NPC] Despejó el carril (NPC detrás/fuera de rango) — CLEAR por FOV+modelo")

        if inject:
          _npc_fov_exit_time = None   # NPC en el carril: resetear timer FOV
          with open(_FAKE_LEAD_FILE, "w") as f:
            f.write(f"{d_rel_inject:.3f} {y_rel:.3f} {v_npc_fwd:.3f}\n")
        elif _npc_cleared:
          # CLEAR activo: suprimir lead de visión mientras el NPC siga en el FOV.
          # Criterio: ángulo del NPC respecto al eje frontal del ego > NPC_CAM_HALF_FOV
          # → cámara frontal no lo puede ver → esperamos NPC_MODEL_FORGET_S para que
          # el modelo olvide el track, y entonces retiramos el CLEAR.
          ego_fwd = np.array([math.cos(env.vehicle.heading_theta), math.sin(env.vehicle.heading_theta)])
          ego_left = np.array([-ego_fwd[1], ego_fwd[0]])
          ego_xy = np.array(env.vehicle.position[:2], dtype=float)
          npc_xy = np.array(roundabout_npc.position[:2], dtype=float)
          delta = npc_xy - ego_xy
          npc_d_fwd = float(np.dot(delta, ego_fwd))    # + = delante
          npc_d_lat = abs(float(np.dot(delta, ego_left)))  # distancia lateral
          npc_dist   = float(np.linalg.norm(delta))

          # Ángulo desde el eje frontal: 0° = recto delante, 90° = lado
          # Si el NPC está detrás (d_fwd < 0) lo tratamos como fuera de FOV siempre.
          if npc_d_fwd > 0:
            cam_angle = math.atan2(npc_d_lat, npc_d_fwd)
          else:
            cam_angle = math.pi  # detrás → definitivamente fuera de FOV

          in_fov = cam_angle < NPC_CAM_HALF_FOV and npc_dist < 60.0

          if in_fov:
            _npc_fov_exit_time = None   # sigue visible, mantener CLEAR
            with open(_FAKE_LEAD_FILE, "w") as f:
              f.write("CLEAR\n")
          else:
            # NPC fuera del FOV: iniciar o comprobar timer de olvido del modelo
            if _npc_fov_exit_time is None:
              _npc_fov_exit_time = time.monotonic()
              print(f"[NPC] Salió del FOV (ángulo={math.degrees(cam_angle):.0f}° "
                    f"d_fwd={npc_d_fwd:.1f}m dist={npc_dist:.1f}m) — "
                    f"esperando {NPC_MODEL_FORGET_S:.1f}s para que modelo olvide")

            elapsed = time.monotonic() - _npc_fov_exit_time
            if elapsed < NPC_MODEL_FORGET_S:
              with open(_FAKE_LEAD_FILE, "w") as f:
                f.write("CLEAR\n")
            else:
              # Modelo ha olvidado el NPC — quitar CLEAR, ACC reanuda
              print(f"[NPC] ACC reanuda (elapsed={elapsed:.1f}s > {NPC_MODEL_FORGET_S:.1f}s)")
              try:
                os.remove(_FAKE_LEAD_FILE)
              except FileNotFoundError:
                pass
              # _npc_cleared permanece True: no volver a inyectar fake_lead
        else:
          try:
            os.remove(_FAKE_LEAD_FILE)
          except FileNotFoundError:
            pass
      # ─────────────────────────────────────────────────────────────────────

      rk.keep_time()

  finally:
    try:
      os.remove(_FAKE_LEAD_FILE)
    except FileNotFoundError:
      pass
