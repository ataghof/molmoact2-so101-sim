"""Scene realism + camera-pose matching for the so101-nexus pick-and-place env.

Matches the sim's look and viewpoints to the real SO-101 rigs MolmoAct2 was trained on:

  - CAMERA: swap the sim's nadir top-down overhead for the oblique third-person views the
    model card shows (a low near-horizontal front camera + a high over-shoulder camera,
    not top-down).
  - WHOLE-SCENE tuning: desaturate the physically-impossible pure colors, wood-texture the
    table, add a light wall background, real material response (specular/shininess/
    reflectance; MuJoCo's OpenGL renderer ignores metallic/roughness), a 3-light rig with
    one soft shadow caster, anti-aliasing, tuned headlight.
  - Arm recolor gold->orange + blue gripper to match the common orange-PLA SO-101 build.

Note from the experiments this repo documents: these changes did NOT move the success
rate (the gap was data, not pixels). They are kept because they cost nothing, make demos
and eval videos easier to read, and keep the sim visually closer to a real rig.

Usage:
    from molmoact2_so101_sim import realism
    realism.install()                      # patch colors + scene builder (before env build)
    env = PickAndPlaceEnv(config=..., render_mode="rgb_array")
    realism.tune_arm(env.unwrapped.model)  # post-compile material tuning
    realism.set_camera(env, "front_low")   # oblique front cam0

The wood texture ships inside the package (`molmoact2_so101_sim/assets/wood_table.png`) and
is loaded through package resources, so it resolves in a non-editable install. Point the
MOLMOACT2_SO101_TEXTURE_DIR environment variable at a directory holding a `wood_table.png`
to override it. The bundled texture is original procedural work (see
tools/generate_wood_texture.py) distributable under this repo's Apache-2.0 license.
"""
from __future__ import annotations

import importlib.resources
import os
import pathlib
import re

import mujoco
import numpy as np

import so101_nexus.constants as C
import so101_nexus.mujoco.pick_and_place as PP
import so101_nexus.object_slots as OS
import so101_nexus.scene as SCN


def _texture_path() -> str:
    """Absolute filesystem path to the wood-table texture (bundled package data by default;
    MOLMOACT2_SO101_TEXTURE_DIR overrides with a directory holding `wood_table.png`)."""
    override = os.environ.get("MOLMOACT2_SO101_TEXTURE_DIR")
    if override:
        return str(pathlib.Path(override) / "wood_table.png")
    return str(importlib.resources.files("molmoact2_so101_sim") / "assets" / "wood_table.png")


TEX_PATH = _texture_path()

# Physically-plausible (desaturated, darker) albedos; pure 1/0 colors read as cartoon.
_REAL_COLORS = {
    "red":    [0.55, 0.13, 0.11, 1.0],
    "orange": [0.85, 0.40, 0.12, 1.0],
    "yellow": [0.78, 0.68, 0.16, 1.0],
    "green":  [0.18, 0.44, 0.20, 1.0],
    "blue":   [0.14, 0.26, 0.52, 1.0],
    "purple": [0.40, 0.18, 0.45, 1.0],
    "black":  [0.08, 0.08, 0.09, 1.0],
    "white":  [0.90, 0.90, 0.88, 1.0],
    "gray":   [0.58, 0.57, 0.55, 1.0],
}

# Realism asset block (textures + materials) injected into the scene XML.
_ASSETS = f"""  <statistic extent="0.8" center="0.18 0 0.12"/>
  <asset>
    <texture name="rk_sky" type="skybox" builtin="gradient" rgb1="0.82 0.82 0.84" rgb2="0.52 0.53 0.55" width="512" height="3072"/>
    <texture name="rk_wood" type="2d" file="{TEX_PATH}"/>
    <material name="rk_table" texture="rk_wood" texuniform="true" texrepeat="3 3" specular="0.2" shininess="0.2" reflectance="0.03"/>
    <material name="rk_wall" rgba="0.87 0.86 0.83 1" specular="0.03" shininess="0.02" reflectance="0.0"/>
    <material name="rk_cube" specular="0.35" shininess="0.4" reflectance="0.02"/>
    <material name="rk_disc" specular="0.25" shininess="0.25" reflectance="0.0"/>
  </asset>
"""

# Tuned <visual>: kill the flat camera fill, AA on, crisp shadows, light haze/room.
_VISUAL = """  <visual>
    <headlight ambient="0.32 0.32 0.33" diffuse="0.22 0.22 0.22" specular="0 0 0"/>
    <quality shadowsize="8192" offsamples="8"/>
    <map shadowclip="1.1" shadowscale="0.65" fogstart="4" fogend="14" haze="0.10"/>
    <rgba haze="0.80 0.80 0.82 1"/>
  </visual>"""

# 3-light studio rig: one warm spot key (soft shadow) + cool fill + rim (no shadow).
_LIGHTS = """    <light name="rk_key" type="spot" pos="0.55 0.45 1.35" dir="-0.35 -0.30 -1"
           diffuse="0.95 0.92 0.85" specular="0.20 0.20 0.20" cutoff="62" exponent="12" castshadow="true"/>
    <light name="rk_fill" type="directional" dir="0.5 0.25 -1" diffuse="0.34 0.36 0.40"
           specular="0 0 0" castshadow="false"/>
    <light name="rk_rim" type="directional" dir="-0.4 -0.7 -0.5" diffuse="0.18 0.18 0.22"
           specular="0.12 0.12 0.12" castshadow="false"/>"""

# A light back wall so the background is a wall, not a void (behind the +X workspace).
_WALL = ('    <geom name="rk_wall" type="box" pos="0.75 0 0.45" size="0.02 1.2 0.6" '
         'material="rk_wall" contype="0" conaffinity="0"/>\n')

_installed = {"done": False}
if not hasattr(OS, "_rk_orig_build"):
    OS._rk_orig_build = OS.build_object_scene_xml


def _realistic_build(*args, **kwargs):
    xml = OS._rk_orig_build(*args, **kwargs)
    xml = xml.replace(SCN.SCENE_VISUAL_XML, _VISUAL, 1)
    xml = xml.replace(SCN.SCENE_LIGHTS_XML, _LIGHTS + "\n" + _WALL, 1)
    xml = xml.replace('<compiler angle="radian"/>', '<compiler angle="radian"/>\n' + _ASSETS, 1)
    # floor plane -> wood table material (drop the flat rgba)
    xml = re.sub(r'(<geom name="floor" type="plane"[^>]*?)\s+rgba="[^"]*"', r'\1 material="rk_table"', xml)
    # cube geom -> add specular material (keeps its desaturated rgba for color)
    xml = re.sub(r'(<geom name="\w+_geom" type="box")', r'\1 material="rk_cube"', xml)
    # target disc -> add specular material
    xml = re.sub(r'(<geom name="target_disc" type="cylinder")', r'\1 material="rk_disc"', xml)
    return xml


def install():
    """Desaturate COLOR_MAP (in place) + monkeypatch the scene builder. Call BEFORE building the env."""
    if _installed["done"]:
        return
    for k, v in _REAL_COLORS.items():
        if k in C.COLOR_MAP:
            C.COLOR_MAP[k] = list(v)  # mutate shared dict -> all importers see it
    OS.build_object_scene_xml = _realistic_build
    PP.build_object_scene_xml = _realistic_build  # pick_and_place imported it by name
    _installed["done"] = True


def tune_arm(model):
    """Post-compile material tuning. MuJoCo's native renderer honors specular/shininess/
    reflectance/rgba only. Recolor gold->orange, gripper->blue, give metal servos a sheen."""
    orange = np.array([0.88, 0.42, 0.12, 1.0])
    blue = np.array([0.15, 0.30, 0.60, 1.0])
    for i in range(model.nmat):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, i) or ""
        rgba = model.mat_rgba[i].copy()
        is_gold = abs(rgba[0] - 1.0) < 0.05 and abs(rgba[1] - 0.82) < 0.05 and abs(rgba[2] - 0.12) < 0.05
        is_black = rgba[0] < 0.15 and rgba[1] < 0.15 and rgba[2] < 0.15
        if "jaw" in name or "gripper" in name or "follower" in name:
            model.mat_rgba[i] = blue
            model.mat_specular[i] = 0.5
            model.mat_shininess[i] = 0.4
        elif is_gold:
            model.mat_rgba[i] = orange
            model.mat_specular[i] = 0.45
            model.mat_shininess[i] = 0.35
            model.mat_reflectance[i] = 0.05
        elif is_black:
            model.mat_specular[i] = 0.5
            model.mat_shininess[i] = 0.5
            model.mat_reflectance[i] = 0.1


# Camera presets: eye + look-at (metres, base frame +X forward), matched to the real
# SO-101 rigs documented for MolmoAct2 (RealSense front + over-shoulder placements).
_CAM_PRESETS = {
    # low, near-horizontal front-corner view (~13 deg down)
    "front_low": (np.array([0.50, -0.25, 0.14]), np.array([0.20, 0.05, 0.04]), 58.0),
    # 40cm high, 27cm from the arm, 45 deg down, ~78 deg HFoV
    "front_45": (np.array([0.60, -0.10, 0.42]), np.array([0.22, 0.0, 0.03]), 60.0),
    # high over-the-shoulder (~48 deg down)
    "shoulder": (np.array([-0.10, -0.05, 0.45]), np.array([0.28, 0.0, 0.02]), 58.0),
}


def set_camera(env, preset="front_low"):
    """Point the sim's overhead observation camera at one of the matched real-rig views."""
    U = env.unwrapped
    eye, target, fovy = _CAM_PRESETS[preset]
    d = eye - target
    dist = float(np.linalg.norm(d))
    cam = U._overhead_obs_cam
    cam.lookat[:] = target
    cam.distance = dist
    cam.elevation = float(-np.degrees(np.arcsin(d[2] / dist)))   # negative = looking down
    cam.azimuth = float(np.degrees(np.arctan2(d[1], d[0])))
    U.model.vis.global_.fovy = float(fovy)
    return preset
