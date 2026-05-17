import math
import os
from multiprocessing import Queue
from metadrive.component.sensors.base_camera import _cuda_enable
from metadrive.component.map.pg_map import MapGenerateMethod
from openpilot.tools.sim.bridge.common import SimulatorBridge
from openpilot.tools.sim.bridge.metadrive.metadrive_common import RGBCameraRoad, RGBCameraWide
from openpilot.tools.sim.bridge.metadrive.metadrive_world import MetaDriveWorld
from openpilot.tools.sim.lib.camerad import W, H

# Activa un NPC circulando en la rotonda.
# Uso: ROUNDABOUT_NPC=1 python tools/sim/run.py
ROUNDABOUT_NPC = int(os.environ.get("ROUNDABOUT_NPC", "0"))


def straight_block(length):
  return {
    "id": "S",
    "pre_block_socket_index": 0,
    "length": length
  }


def roundabout_block(inner_radius=30, exit_radius=15, angle=60):
  """
  Bloque de rotonda nativo de MetaDrive (ID="O").

  Parámetros:
    inner_radius : radio del anillo interior de la rotonda (metros)
                   Por defecto 20 en MetaDrive — subimos a 30 para que
                   el coche tenga espacio suficiente para circular.
    exit_radius  : radio de las curvas de entrada/salida (metros)
                   Por defecto igual al inner_radius en el bloque nativo.
    angle        : ángulo de las curvas de entrada/salida (grados)
                   Por defecto 60 en MetaDrive — lo mantenemos.

  SOCKET_NUM=3 → la rotonda tiene 4 conexiones:
    - socket 0: la que viene del bloque anterior (entrada principal)
    - socket 1, 2, 3: salidas disponibles
  Usamos pre_block_socket_index=0 para conectar con la recta de entrada.
  La salida la cogemos por socket 1 (enfrente de la entrada).
  """
  return {
    "id": "O",
    "pre_block_socket_index": 0,
    "inner_radius": inner_radius,
    "exit_radius": exit_radius,
    "angle": angle,
  }


def create_map():
  """
  Mapa con rotonda inspirada en la rotonda de la Universidad Europea
  de Villaviciosa de Odón:
    - Recta de aproximación larga (100m) para que el coche llegue
      con tiempo de estabilizarse
    - Rotonda con radio interior de 30m — suficientemente grande para
      que el coche circule sin salirse
    - Recta de salida (80m) tras pasar la rotonda

  El coche entra por la recta, da la vuelta en la rotonda y sale
  por la salida principal (socket 1, enfrente de la entrada).
  """
  return dict(
    type=MapGenerateMethod.PG_MAP_FILE,
    lane_num=2,       # 2 carriles — suficiente para la rotonda
    lane_width=4.0,   # carriles más anchos que el default (3.5)
                      # para dar más margen en las curvas de la rotonda
    config=[
      None,
      straight_block(100),      # recta de aproximación
      roundabout_block(
        inner_radius=30,        # anillo interior amplio
        exit_radius=15,         # curvas de entrada/salida suaves
        angle=60,               # ángulo default de MetaDrive
      ),
      straight_block(80),       # recta de salida
    ]
  )


class MetaDriveBridge(SimulatorBridge):
  TICKS_PER_FRAME = 5

  def __init__(self, dual_camera, high_quality, test_duration=math.inf, test_run=False):
    super().__init__(dual_camera, high_quality)
    self.should_render = False
    self.test_run      = test_run
    self.test_duration = test_duration if self.test_run else math.inf

  def spawn_world(self, queue: Queue):
    sensors = {
      "rgb_road": (RGBCameraRoad, W, H,)
    }
    if self.dual_camera:
      sensors["rgb_wide"] = (RGBCameraWide, W, H)

    config = dict(
      use_render=self.should_render,
      vehicle_config=dict(
        enable_reverse=False,
        render_vehicle=False,
        image_source="rgb_road",
        show_navi_mark=False,
        show_dest_mark=False,
        show_line_to_dest=False,
        show_line_to_navi_mark=False,
      ),
      sensors=sensors,
      image_on_cuda=_cuda_enable,
      image_observation=True,
      interface_panel=[],
      out_of_route_done=False,
      on_continuous_line_done=False,
      crash_vehicle_done=False,
      crash_object_done=False,
      arrive_dest_done=False,
      traffic_density=0.0,
      map_config=create_map(),
      decision_repeat=1,
      physics_world_step_size=self.TICKS_PER_FRAME / 100,
      preload_models=False,
      show_logo=False,
      anisotropic_filtering=False,
    )

    return MetaDriveWorld(queue, config, self.test_duration, self.test_run, self.dual_camera)