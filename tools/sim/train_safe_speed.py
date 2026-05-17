#!/usr/bin/env python3
"""
train_safe_speed.py
===================
Entrena un MLP de velocidad segura usando velocidades REALES de conductores
en rotondas (openDD) como ground truth.

Contexto de uso:
  El modelo se activa cuando un algoritmo externo detecta peligro (ext_rec=0).
  En ese momento el coche va a cierta velocidad y hay que decidir a qué
  velocidad objetivo reducir. El modelo aprende de cómo conductores reales
  gestionaron exactamente esa situación.

Diferencias clave respecto al script anterior:
  - Label = velocidad real del conductor LOOKAHEAD_S segundos después
  - Solo se usan momentos de aproximación/entrada (el coche frena o va a entrar)
  - Añade a_ego como feature
  - Sin fórmulas físicas para el label

Features: [v_ego_ms, lead_dist_m, lead_speed_ms, has_lead, a_ego_ms2]
Label:    v_real_futura_kph  (lo que hizo el conductor real)

Uso:
  python train_safe_speed.py [--plot] [--no-download] [--full]
"""

import argparse
import sqlite3
import sys
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path

# ── Constantes ────────────────────────────────────────────────────────────
MS_TO_KPH       = 3.6
SPEED_FLOOR_KPH = 5.0
LOOKAHEAD_S     = 2.0    # segundos: label = velocidad del conductor en t+2s
MAX_SPEED_MS    = 22.0   # descartar valores irreales (>79 km/h)

# Filtro de "situación de peligro/aproximación":
# Solo usamos momentos en que el vehículo está decelerando O va a más de
# MIN_APPROACH_KPH (se acerca a la rotonda con velocidad relevante).
MIN_APPROACH_KPH = 10.0  # km/h — descartar velocidades muy bajas
MAX_APPROACH_KPH = 80.0  # km/h — descartar autopista

# ── URLs openDD ───────────────────────────────────────────────────────────
OPENDD_BASE  = "https://data.l3pilot.eu/OpenDD"
OPENDD_FILES = {
  "example": (f"{OPENDD_BASE}/opendd_v3-example_data.zip", "~14 MB"),
  "rdb1":    (f"{OPENDD_BASE}/opendd_v3-rdb1.zip",         "~733 MB"),
}


# ═════════════════════════════════════════════════════════════════════════
# DESCARGA
# ═════════════════════════════════════════════════════════════════════════

def download_opendd(dest_dir: Path, full: bool = False) -> bool:
  dest_dir.mkdir(parents=True, exist_ok=True)
  key = "rdb1" if full else "example"
  url, size = OPENDD_FILES[key]
  zip_path  = dest_dir / f"opendd_{key}.zip"

  if zip_path.exists():
    print(f"[DOWNLOAD] {zip_path.name} ya existe, saltando.")
  else:
    print(f"[DOWNLOAD] {zip_path.name} ({size})...", end=" ", flush=True)
    try:
      req = urllib.request.Request(url, headers={"User-Agent": "openpilot-trainer"})
      with urllib.request.urlopen(req, timeout=180) as r, open(zip_path, "wb") as f:
        f.write(r.read())
      print("OK")
    except Exception as e:
      print(f"ERROR — {e}")
      return False

  print(f"[DOWNLOAD] Descomprimiendo {zip_path.name}...", end=" ", flush=True)
  try:
    with zipfile.ZipFile(zip_path, "r") as z:
      z.extractall(dest_dir)
    print("OK")
    return True
  except Exception as e:
    print(f"ERROR — {e}")
    return False


# ═════════════════════════════════════════════════════════════════════════
# EXTRACCIÓN DE FEATURES Y LABELS REALES
# ═════════════════════════════════════════════════════════════════════════

def _find_lead(snapshot: pd.DataFrame, max_dist_m: float = 80.0,
               cone_deg: float = 35.0) -> dict:
  """
  Para cada vehículo en un snapshot, encuentra su líder (cono frontal).
  Devuelve dict: OBJID → (lead_dist_m, lead_speed_ms, has_lead)
  """
  xs      = snapshot["UTM_X"].values
  ys      = snapshot["UTM_Y"].values
  vs      = snapshot["V"].values
  angles  = snapshot["UTM_ANGLE"].values
  lengths = snapshot["LENGTH"].values
  objids  = snapshot["OBJID"].values
  n       = len(snapshot)
  cos_th  = np.cos(np.radians(cone_deg))
  fwd_x   = np.cos(angles)
  fwd_y   = np.sin(angles)

  result = {}
  for i in range(n):
    dx   = xs - xs[i]
    dy   = ys - ys[i]
    dist = np.hypot(dx, dy)
    valid = (dist > 0.3) & (dist < max_dist_m)
    if not valid.any():
      result[objids[i]] = (80.0, vs[i], 0)
      continue
    dot     = np.where(dist > 0, (dx * fwd_x[i] + dy * fwd_y[i]) / dist, -1.0)
    in_cone = valid & (dot >= cos_th)
    if not in_cone.any():
      result[objids[i]] = (80.0, vs[i], 0)
      continue
    d_cone = dist[in_cone]
    v_cone = vs[in_cone]
    l_cone = lengths[in_cone]
    idx    = np.argmin(d_cone)
    bumper = max(0.0, d_cone[idx] - lengths[i] / 2.0 - l_cone[idx] / 2.0)
    result[objids[i]] = (bumper, v_cone[idx], 1)

  return result


def load_opendd_real_labels(data_dir: Path, max_recordings: int = 15,
                             lookahead_s: float = LOOKAHEAD_S) -> pd.DataFrame | None:
  """
  Carga openDD y genera el dataset con labels REALES.

  Para cada vehículo, para cada timestep t:
    - Calcula a_ego como derivada de V (aceleración real del conductor)
    - Busca el líder en ese snapshot
    - Busca la velocidad del mismo vehículo en t + lookahead_s  ← LABEL REAL
    - Solo guarda momentos de aproximación (filtro de peligro)
  """
  sqlite_files = sorted(data_dir.glob("**/rdb*_*.sqlite"))
  if not sqlite_files:
    return None

  classes_motor = {"Car", "Van", "Truck", "Motorcycle"}
  all_dfs = []

  for db_path in sqlite_files:
    conn   = sqlite3.connect(str(db_path))
    tables = pd.read_sql(
      "SELECT name FROM sqlite_master WHERE type='table'", conn
    )["name"].tolist()
    conn.close()

    proc = min(len(tables), max_recordings)
    print(f"[OPENDD] {db_path.name}: procesando {proc}/{len(tables)} recordings.")

    for tbl in tables[:proc]:
      conn   = sqlite3.connect(str(db_path))
      df_raw = pd.read_sql(
        f"SELECT TIMESTAMP, OBJID, UTM_X, UTM_Y, UTM_ANGLE, V, LENGTH, CLASS "
        f"FROM '{tbl}'", conn
      )
      conn.close()

      df_raw = df_raw[df_raw["CLASS"].isin(classes_motor)].copy()
      if df_raw.empty:
        continue

      df_raw = df_raw.sort_values(["OBJID", "TIMESTAMP"]).reset_index(drop=True)

      # ── Detectar paso de tiempo (fps) ────────────────────────────────
      sample_obj = df_raw["OBJID"].iloc[0]
      sample_ts  = df_raw[df_raw["OBJID"] == sample_obj]["TIMESTAMP"].values
      if len(sample_ts) < 2:
        continue
      ts_step    = np.median(np.diff(sample_ts))   # paso mediano
      if ts_step <= 0:
        continue
      lookahead_steps = max(1, int(round(lookahead_s / ts_step)))

      # ── Aceleración real por vehículo (derivada central de V) ────────
      df_raw["a_ego"] = (
        df_raw.groupby("OBJID")["V"]
        .transform(lambda s: pd.Series(
          np.gradient(s.values, ts_step), index=s.index
        ))
      )

      # ── Índice de líder por timestamp ────────────────────────────────
      ts_unique   = df_raw["TIMESTAMP"].unique()
      lead_lookup = {}           # (ts, objid) → (dist, speed, has)
      for ts in ts_unique:
        snap = df_raw[df_raw["TIMESTAMP"] == ts]
        if len(snap) < 1:
          continue
        res = _find_lead(snap)
        for oid, vals in res.items():
          lead_lookup[(ts, oid)] = vals

      # ── Label real: V del mismo vehículo lookahead_steps más tarde ───
      # Construir mapa (objid, ts_index) → V para búsqueda rápida
      df_raw["ts_idx"] = df_raw.groupby("OBJID").cumcount()
      v_map = {}  # (objid, ts_idx) → V
      for row in df_raw[["OBJID", "ts_idx", "V"]].itertuples(index=False):
        v_map[(row.OBJID, row.ts_idx)] = row.V

      records = []
      for row in df_raw.itertuples(index=False):
        # Buscar la velocidad futura del mismo vehículo
        v_future = v_map.get((row.OBJID, row.ts_idx + lookahead_steps))
        if v_future is None:
          continue   # sin datos futuros → saltar

        lead = lead_lookup.get((row.TIMESTAMP, row.OBJID), (80.0, row.V, 0))

        records.append({
          "v_ego_ms":      float(row.V),
          "a_ego_ms2":     float(row.a_ego),
          "lead_dist_m":   float(lead[0]),
          "lead_speed_ms": float(lead[1]),
          "has_lead":      float(lead[2]),
          "v_future_ms":   float(v_future),   # ← LABEL REAL
        })

      if not records:
        continue

      chunk = pd.DataFrame(records)

      # ── Filtro: solo situaciones de peligro/reducción ────────────────
      # Solo guardamos momentos en que el conductor REDUJO velocidad o la mantuvo.
      # Si el conductor aceleró (v_future > v_ego), ese momento no es útil para
      # un modelo que se activa ante peligro (el peligro implica reducir, no acelerar).
      v_kph = chunk["v_ego_ms"] * MS_TO_KPH
      is_danger_relevant = (
        (v_kph >= MIN_APPROACH_KPH) &
        (v_kph <= MAX_APPROACH_KPH) &
        (chunk["v_ego_ms"] < MAX_SPEED_MS) &
        (chunk["v_future_ms"] >= 0.0) &
        (chunk["v_future_ms"] < MAX_SPEED_MS) &
        (chunk["v_future_ms"] <= chunk["v_ego_ms"])   # el conductor redujo o mantuvo
      )
      chunk = chunk[is_danger_relevant].reset_index(drop=True)

      if len(chunk) > 0:
        all_dfs.append(chunk)
        print(f"[OPENDD]   {tbl}: {len(chunk)} muestras de aproximación.")

  if not all_dfs:
    return None

  df = pd.concat(all_dfs, ignore_index=True)
  print(f"\n[OPENDD] Total antes de limpieza: {len(df)} muestras")

  # Limpieza final
  before = len(df)
  df = df[
    (df["v_ego_ms"]      > 0.5) &
    (df["lead_dist_m"]   > 0.5) &
    (df["lead_dist_m"]   < 120.0) &
    (df["lead_speed_ms"] >= 0.0) &
    (df["lead_speed_ms"] < MAX_SPEED_MS) &
    (df["a_ego_ms2"]     > -8.0) &   # descartar frenadas de emergencia extremas
    (df["a_ego_ms2"]     < 4.0)      # descartar aceleraciones extremas
  ].reset_index(drop=True)

  print(f"[OPENDD] Tras limpieza: {len(df)} muestras ({before - len(df)} descartadas)\n")
  return df


# ═════════════════════════════════════════════════════════════════════════
# ANÁLISIS DEL DATASET
# ═════════════════════════════════════════════════════════════════════════

def describe_dataset(df: pd.DataFrame):
  v_ego_kph    = df["v_ego_ms"]      * MS_TO_KPH
  v_future_kph = df["v_future_ms"]   * MS_TO_KPH
  reduccion    = v_ego_kph - v_future_kph

  print("=" * 55)
  print("  ANÁLISIS DEL DATASET")
  print("=" * 55)
  print(f"  Total muestras:         {len(df):>8,}")
  print(f"  Con líder:              {df['has_lead'].sum():>8,.0f}  ({df['has_lead'].mean()*100:.1f}%)")
  print(f"  Sin líder:              {(1-df['has_lead']).sum():>8,.0f}  ({(1-df['has_lead']).mean()*100:.1f}%)")
  print()
  print(f"  v_ego  media:           {v_ego_kph.mean():>7.2f} km/h   std={v_ego_kph.std():.2f}")
  print(f"  v_ego  rango:           [{v_ego_kph.min():.1f}, {v_ego_kph.max():.1f}] km/h")
  print()
  print(f"  v_label media:          {v_future_kph.mean():>7.2f} km/h   std={v_future_kph.std():.2f}")
  print(f"  v_label rango:          [{v_future_kph.min():.1f}, {v_future_kph.max():.1f}] km/h")
  print()
  print(f"  Reducción media:        {reduccion.mean():>7.2f} km/h")
  print(f"  Casos sin reducción:    {(reduccion <= 0).sum():>8,}  ({(reduccion <= 0).mean()*100:.1f}%)")
  print(f"  a_ego media:            {df['a_ego_ms2'].mean():>7.3f} m/s²")
  print("=" * 55)
  print()


# ═════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ═════════════════════════════════════════════════════════════════════════

def train_mlp(X: np.ndarray, y: np.ndarray):
  try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
  except ImportError:
    print("[ERROR] sklearn no instalado. Ejecuta: uv sync --extra dev")
    sys.exit(1)

  X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, random_state=42)
  sc = StandardScaler()
  X_tr_sc = sc.fit_transform(X_tr)
  X_te_sc = sc.transform(X_te)

  print(f"[TRAIN] Train: {len(X_tr):,}   Test: {len(X_te):,}")
  print(f"[TRAIN] Features: {X.shape[1]}   "
        f"[v_ego, a_ego, lead_dist, lead_speed, has_lead]")
  print(f"[TRAIN] Entrenando MLP (5→32→16→1, relu, adam)...", end=" ", flush=True)

  mlp = MLPRegressor(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    learning_rate_init=1e-3,
    max_iter=800,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=30,
    random_state=42,
    verbose=False,
  )
  mlp.fit(X_tr_sc, y_tr)
  print("OK")

  y_pred = mlp.predict(X_te_sc)
  rmse   = float(np.sqrt(mean_squared_error(y_te, y_pred)))
  mae    = float(mean_absolute_error(y_te, y_pred))
  r2     = float(r2_score(y_te, y_pred))

  # ── Métricas de seguridad ────────────────────────────────────────────
  # ¿Cuántas veces el modelo predice MÁS velocidad que v_ego? (nunca debería en peligro)
  v_ego_test_kph = X_te[:, 0] * MS_TO_KPH
  over_ego       = np.sum(y_pred > v_ego_test_kph)
  over_ego_pct   = over_ego / len(y_pred) * 100

  # ¿Cuántas veces el modelo predice menos del SPEED_FLOOR?
  below_floor    = np.sum(y_pred < SPEED_FLOOR_KPH)

  print()
  print("=" * 55)
  print("  MÉTRICAS DE RENDIMIENTO")
  print("=" * 55)
  print(f"  RMSE:                   {rmse:>7.3f} km/h")
  print(f"  MAE:                    {mae:>7.3f} km/h")
  print(f"  R²:                     {r2:>7.4f}")
  print(f"  Iteraciones:            {mlp.n_iter_:>7}")
  print()
  print("  MÉTRICAS DE SEGURIDAD")
  print(f"  Predicciones > v_ego:   {over_ego:>7,}  ({over_ego_pct:.1f}%) ← debería ser bajo")
  print(f"  Predicciones < floor:   {below_floor:>7,}")
  print("=" * 55)
  print()

  # ── Análisis por cuartil de velocidad ───────────────────────────────
  print("  ERROR POR RANGO DE VELOCIDAD ACTUAL")
  print(f"  {'Rango (km/h)':<20} {'N':>6}  {'MAE':>8}  {'RMSE':>8}")
  print("  " + "-" * 46)
  quartiles = [0, 20, 35, 50, 999]
  for lo, hi in zip(quartiles[:-1], quartiles[1:]):
    mask = (v_ego_test_kph >= lo) & (v_ego_test_kph < hi)
    if mask.sum() == 0:
      continue
    mae_q  = float(np.mean(np.abs(y_pred[mask] - y_te[mask])))
    rmse_q = float(np.sqrt(np.mean((y_pred[mask] - y_te[mask])**2)))
    label  = f"[{lo}-{hi}) km/h" if hi < 999 else f"≥{lo} km/h"
    print(f"  {label:<20} {mask.sum():>6,}  {mae_q:>7.3f}  {rmse_q:>7.3f}")
  print()

  return mlp, sc, X_te, y_te, y_pred


# ═════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN
# ═════════════════════════════════════════════════════════════════════════

def export_weights(mlp, scaler, out_path: Path):
  weights = {}
  for i, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
    weights[f"W{i}"] = W.astype(np.float32)
    weights[f"b{i}"] = b.astype(np.float32)
  weights["in_mean"]    = scaler.mean_.astype(np.float32)
  weights["in_std"]     = scaler.scale_.astype(np.float32)
  weights["out_mean"]   = np.float32(0.0)
  weights["out_std"]    = np.float32(1.0)
  weights["n_features"] = np.int32(mlp.coefs_[0].shape[0])

  out_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(str(out_path), **weights)

  # Verificación: [v=10m/s, a=-1m/s², dist=20m, v_lead=8m/s, has_lead=1]
  x = np.array([10.0, -1.0, 20.0, 8.0, 1.0], dtype=np.float32)
  x_n = (x - weights["in_mean"]) / (weights["in_std"] + 1e-8)
  n_layers = sum(1 for k in weights if k.startswith("W"))
  for i in range(n_layers):
    x_n = x_n @ weights[f"W{i}"] + weights[f"b{i}"]
    if i < n_layers - 1:
      x_n = np.maximum(0.0, x_n)

  print(f"[EXPORT] {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")
  print(f"[VERIFY] [10 m/s, -1 m/s², 20m, 8 m/s, líder=sí] → v_safe = {float(x_n[0]):.2f} km/h")


# ═════════════════════════════════════════════════════════════════════════
# GRÁFICAS
# ═════════════════════════════════════════════════════════════════════════

def plot_label_distribution(v_ego_ms: np.ndarray, y_real: np.ndarray, out_dir: Path):
  """
  Ilustra el problema del suelo de velocidad en la versión anterior del entrenamiento.

  Versión anterior: label = max(v_ego_kph * 0.4, 7.0)
    → para v_ego entre 5-18 km/h el resultado siempre era < 7, y el floor
      recortaba todo al mismo valor. El modelo aprendía a predecir siempre 7.

  Versión actual: label = v_real_futura (openDD) clipeada a ≥ 5 km/h
    → distribución proporcional a la velocidad real del conductor.
  """
  try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
  except ImportError:
    print("[WARN] matplotlib no disponible — saltando gráficas.")
    return

  OLD_FACTOR = 0.4
  OLD_FLOOR  = 7.0
  v_ego_kph  = v_ego_ms * MS_TO_KPH

  # Reconstruir etiquetas de la versión anterior
  y_old = np.maximum(v_ego_kph * OLD_FACTOR, OLD_FLOOR).astype(np.float32)

  # Muestras que la versión anterior recortaba al floor (zona problemática)
  old_clipped_mask = (v_ego_kph * OLD_FACTOR) < OLD_FLOOR
  n_old_clipped    = int(old_clipped_mask.sum())
  pct_old_clipped  = n_old_clipped / len(y_old) * 100

  bins = np.linspace(0, MAX_APPROACH_KPH, 65)

  fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
  fig.suptitle(
    "Distribución de etiquetas — antes y después del ajuste del suelo de velocidad",
    fontsize=12, fontweight="bold",
  )

  # ── Panel izquierdo: versión anterior ──────────────────────────────────
  ax = axes[0]
  ax.hist(y_old, bins=bins, color="#d9534f", alpha=0.80, edgecolor="none",
          label="label = max(v_ego × 0.4, 7)")
  ax.axvline(OLD_FLOOR, color="darkred", ls="--", lw=2.0,
             label=f"Floor anterior = {OLD_FLOOR} km/h")
  ax.axvline(float(np.mean(y_old)), color="orange", ls="--", lw=1.3,
             label=f"Media = {np.mean(y_old):.1f} km/h")
  # Flecha señalando el spike
  spike_height = int(old_clipped_mask.sum())
  ax.annotate(
    f"{n_old_clipped:,} muestras ({pct_old_clipped:.0f}%)\nrecortadas al floor",
    xy=(OLD_FLOOR, spike_height * 0.92),
    xytext=(OLD_FLOOR + 6, spike_height * 0.75),
    arrowprops=dict(arrowstyle="->", color="darkred", lw=1.4),
    fontsize=8.5, color="darkred",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
  )
  stats_old = (
    f"N = {len(y_old):,}\n"
    f"Media   = {np.mean(y_old):.2f} km/h\n"
    f"Mediana = {np.median(y_old):.2f} km/h\n"
    f"Std     = {np.std(y_old):.2f} km/h\n"
    f"Único valor dominante: {OLD_FLOOR} km/h"
  )
  ax.text(0.97, 0.97, stats_old, transform=ax.transAxes,
          fontsize=8.5, va="top", ha="right",
          bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))
  ax.set_xlabel("v_safe label (km/h)")
  ax.set_ylabel("Frecuencia")
  ax.set_title("ANTES  —  v_ego × 0.4  |  floor = 7 km/h\n"
               r"$\bf{Problema:\ el\ floor\ domina\ la\ distribución}$",
               fontsize=10)
  ax.legend(fontsize=8)
  ax.grid(True, alpha=0.3)

  # ── Panel derecho: versión actual ──────────────────────────────────────
  ax = axes[1]
  ax.hist(y_real, bins=bins, color="#5cb85c", alpha=0.80, edgecolor="none",
          label="label = v_real_futura (openDD)")
  ax.axvline(SPEED_FLOOR_KPH, color="darkgreen", ls="--", lw=2.0,
             label=f"Floor actual = {SPEED_FLOOR_KPH} km/h")
  ax.axvline(float(np.mean(y_real)), color="orange", ls="--", lw=1.3,
             label=f"Media = {np.mean(y_real):.1f} km/h")
  stats_new = (
    f"N = {len(y_real):,}\n"
    f"Media   = {np.mean(y_real):.2f} km/h\n"
    f"Mediana = {np.median(y_real):.2f} km/h\n"
    f"Std     = {np.std(y_real):.2f} km/h\n"
    f"Distribución continua y proporcional"
  )
  ax.text(0.97, 0.97, stats_new, transform=ax.transAxes,
          fontsize=8.5, va="top", ha="right",
          bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))
  ax.set_xlabel("v_safe label (km/h)")
  ax.set_ylabel("Frecuencia")
  ax.set_title("DESPUÉS  —  v_real_futura openDD  |  floor = 5 km/h\n"
               r"$\bf{Etiquetas\ proporcionales\ a\ la\ velocidad\ real}$",
               fontsize=10)
  ax.legend(fontsize=8)
  ax.grid(True, alpha=0.3)

  fig.text(
    0.5, 0.005,
    f"Factor corregido: 0.4 → 0.80  |  Floor corregido: {OLD_FLOOR} → {SPEED_FLOOR_KPH} km/h  |  "
    f"Labels reemplazadas por velocidades reales de conductores (openDD)",
    ha="center", fontsize=8.5, color="#555",
  )

  plt.tight_layout(rect=[0, 0.04, 1, 1])
  path = out_dir / "label_distribution.png"
  plt.savefig(str(path), dpi=140)
  plt.close(fig)
  print(f"[PLOT] Distribución de etiquetas → {path}")
  print(f"[PLOT] Versión anterior: {n_old_clipped:,} muestras ({pct_old_clipped:.1f}%) "
        f"colapsadas al floor {OLD_FLOOR} km/h")


def plot_real_vs_predicted(y_te: np.ndarray, y_pred: np.ndarray, out_dir: Path):
  """Scatter real vs predicho para el documento — figura de cierre del MLP v2."""
  try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
  except ImportError:
    print("[WARN] matplotlib/sklearn no disponible — saltando gráficas.")
    return

  rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
  mae  = float(mean_absolute_error(y_te, y_pred))
  r2   = float(r2_score(y_te, y_pred))

  lim_min = min(y_te.min(), y_pred.min()) - 1.0
  lim_max = max(y_te.max(), y_pred.max()) + 1.0

  fig, ax = plt.subplots(figsize=(6, 6))

  # Densidad por hexbin para no saturar con 6k puntos
  hb = ax.hexbin(y_te, y_pred, gridsize=55, cmap="Blues", mincnt=1,
                 extent=[lim_min, lim_max, lim_min, lim_max])
  cb = fig.colorbar(hb, ax=ax, pad=0.02)
  cb.set_label("Muestras por celda", fontsize=9)

  # Línea ideal y=x
  ax.plot([lim_min, lim_max], [lim_min, lim_max],
          color="#d9534f", lw=1.8, ls="--", label="Predicción perfecta (y = x)")

  # Banda ±2 km/h
  ax.fill_between([lim_min, lim_max],
                  [lim_min - 2, lim_max - 2],
                  [lim_min + 2, lim_max + 2],
                  color="#d9534f", alpha=0.08, label="Banda ±2 km/h")

  within_2 = float(np.mean(np.abs(y_pred - y_te) <= 2.0) * 100)

  metrics_txt = (
    f"RMSE = {rmse:.3f} km/h\n"
    f"MAE  = {mae:.3f} km/h\n"
    f"R²   = {r2:.4f}\n"
    f"N    = {len(y_te):,}\n"
    f"Dentro de ±2 km/h: {within_2:.1f}%"
  )
  ax.text(0.04, 0.96, metrics_txt, transform=ax.transAxes,
          fontsize=9, va="top", ha="left", family="monospace",
          bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#ccc", alpha=0.9))

  ax.set_xlim(lim_min, lim_max)
  ax.set_ylim(lim_min, lim_max)
  ax.set_aspect("equal")
  ax.set_xlabel("v_safe real — conductor openDD (km/h)", fontsize=10)
  ax.set_ylabel("v_safe predicho — MLP v2 (km/h)", fontsize=10)
  ax.set_title("MLP v2 — Valores reales vs predichos\n"
               "Labels: velocidad real del conductor 2 s después (openDD)",
               fontsize=10)
  ax.legend(fontsize=8.5, loc="lower right")
  ax.grid(True, alpha=0.25)
  ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
  ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

  plt.tight_layout()
  path = out_dir / "real_vs_predicted_mlp_v2.png"
  plt.savefig(str(path), dpi=150)
  plt.close(fig)
  print(f"[PLOT] Real vs predicho MLP v2 → {path}")
  print(f"[PLOT] RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.4f}  "
        f"Dentro±2km/h={within_2:.1f}%")


def plot_results(X_te, y_te, y_pred, out_dir: Path):
  try:
    import matplotlib.pyplot as plt
  except ImportError:
    print("[WARN] matplotlib no disponible — saltando gráficas.")
    return

  v_ego_kph = X_te[:, 0] * MS_TO_KPH
  errors    = y_pred - y_te
  rmse      = float(np.sqrt(np.mean(errors**2)))

  fig, axes = plt.subplots(1, 3, figsize=(16, 5))
  fig.suptitle("MLP Velocidad Segura — Entrenado con Conductores Reales (openDD)", fontsize=13)

  # ── Real vs Predicho ────────────────────────────────────────────────
  ax = axes[0]
  lim = [SPEED_FLOOR_KPH, max(y_te.max(), y_pred.max()) + 2]
  ax.scatter(y_te, y_pred, alpha=0.15, s=3, c="steelblue")
  ax.plot(lim, lim, "r--", lw=1.5, label="Ideal")
  ax.set_xlabel("v_safe real — conductor (km/h)")
  ax.set_ylabel("v_safe predicho — modelo (km/h)")
  ax.set_title("Real vs Predicho")
  ax.legend(); ax.grid(True, alpha=0.3)

  # ── Distribución de errores ─────────────────────────────────────────
  ax = axes[1]
  ax.hist(errors, bins=60, edgecolor="none", color="steelblue", alpha=0.85)
  ax.axvline(0, color="red", ls="--", lw=1.5)
  ax.axvline(errors.mean(), color="orange", ls="--", lw=1.2,
             label=f"Media={errors.mean():.2f}")
  ax.set_xlabel("Error (km/h)")
  ax.set_title(f"Distribución de errores — RMSE={rmse:.3f} km/h")
  ax.legend(); ax.grid(True, alpha=0.3)

  # ── Velocidad predicha vs v_ego (el modelo siempre debe reducir) ────
  ax = axes[2]
  ax.scatter(v_ego_kph, y_pred, alpha=0.1, s=2, c="steelblue", label="Predicho")
  ax.scatter(v_ego_kph, y_te,   alpha=0.1, s=2, c="green",     label="Real")
  ax.plot([0, 80], [0, 80], "r--", lw=1.2, label="v_safe = v_ego")
  ax.set_xlabel("v_ego (km/h)")
  ax.set_ylabel("v_safe (km/h)")
  ax.set_title("v_safe vs v_ego — el modelo debe estar bajo la línea roja")
  ax.legend(); ax.grid(True, alpha=0.3)

  plt.tight_layout()
  path = out_dir / "safe_speed_validation.png"
  plt.savefig(str(path), dpi=130)
  print(f"[PLOT] → {path}")


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
  parser = argparse.ArgumentParser(
    description="Entrena MLP velocidad segura con labels reales (openDD)"
  )
  parser.add_argument("--out",          type=Path,
                      default=Path("selfdrive/controls/lib/mlp_danger_weights.npz"))
  parser.add_argument("--data-dir",     type=Path, default=Path("tools/sim/data/opendd"))
  parser.add_argument("--no-download",  action="store_true")
  parser.add_argument("--full",         action="store_true",
                      help="Descargar rdb1 completo (733 MB) en vez del ejemplo (14 MB)")
  parser.add_argument("--lookahead",    type=float, default=LOOKAHEAD_S,
                      help=f"Segundos de anticipación para el label (default {LOOKAHEAD_S})")
  parser.add_argument("--plot",         action="store_true")
  args = parser.parse_args()

  print("=" * 55)
  print("  MLP Safe Speed Trainer — openDD real labels")
  print("=" * 55)
  print()

  # ── Datos ────────────────────────────────────────────────────────────
  has_sqlite = any(args.data_dir.glob("**/*.sqlite"))
  if not has_sqlite and not args.no_download:
    print("[INFO] Datos no encontrados — descargando openDD...")
    if not download_opendd(args.data_dir, full=args.full):
      print("[ERROR] No se pudieron descargar los datos.")
      sys.exit(1)

  print(f"[INFO] Cargando openDD desde {args.data_dir} (lookahead={args.lookahead}s)...")
  df = load_opendd_real_labels(args.data_dir, lookahead_s=args.lookahead)
  if df is None or len(df) < 100:
    print("[ERROR] No hay suficientes datos. Asegúrate de tener openDD en --data-dir")
    sys.exit(1)

  describe_dataset(df)

  # ── Features y label ────────────────────────────────────────────────
  # Label: velocidad futura real en km/h (igual que el modelo anterior para compatibilidad)
  X = df[["v_ego_ms", "a_ego_ms2", "lead_dist_m", "lead_speed_ms", "has_lead"]].values.astype(np.float32)
  y = np.clip((df["v_future_ms"] * MS_TO_KPH).values.astype(np.float32), SPEED_FLOOR_KPH, 200.0)

  # ── Distribución de etiquetas (siempre) ──────────────────────────────
  plot_label_distribution(X[:, 0], y, args.out.parent)

  # ── Entrenamiento ────────────────────────────────────────────────────
  mlp, sc, X_te, y_te, y_pred = train_mlp(X, y)

  # ── Gráficas ─────────────────────────────────────────────────────────
  plot_real_vs_predicted(y_te, y_pred, args.out.parent)
  if args.plot:
    plot_results(X_te, y_te, y_pred, args.out.parent)

  # ── Exportar ─────────────────────────────────────────────────────────
  export_weights(mlp, sc, args.out)
  print()
  print(f"[OK] Pesos guardados en {args.out}")
  print(f"     Para activar en el COMMA:")
  print(f"     scp {args.out} comma@comma:/data/params/mlp_danger_weights.npz")
  print()


if __name__ == "__main__":
  main()
