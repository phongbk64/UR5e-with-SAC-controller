"""Replay the MATLAB SAC joint trajectory in the original UR5e MuJoCo model."""

from __future__ import annotations

import argparse
import math
import threading
import time
import tkinter as tk
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_XML = (HERE.parent / "universal_robots_ur5e" / "scene.xml").resolve()
DEFAULT_XLSX = Path(
    r"C:\Users\admin\Desktop\XIN VIEC\PROJECT\MATLAB\final_training\joint_trajectory.xlsx"
)
GOAL_POSITION = np.array([-0.17003438, -0.63091222, 0.40033002])
GOAL_DIRECTION = np.array([0.29619813, -0.17101007, -0.93969262])
POSITION_TOLERANCE = 0.005
ANGLE_TOLERANCE_DEG = 1.0


def read_joint_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read Time_s and q1_rad...q6_rad using only Python's standard library."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as book:
        shared = []
        if "xl/sharedStrings.xml" in book.namelist():
            root = ET.fromstring(book.read("xl/sharedStrings.xml"))
            shared = ["".join(node.itertext()) for node in root.findall(f"{ns}si")]
        sheet_name = next(
            name for name in book.namelist() if name.startswith("xl/worksheets/sheet")
        )
        sheet = ET.fromstring(book.read(sheet_name))
        rows = []
        for row in sheet.findall(f".//{ns}row"):
            values = []
            for cell in row.findall(f"{ns}c"):
                kind = cell.get("t")
                value = cell.find(f"{ns}v")
                if kind == "inlineStr":
                    text = cell.find(f".//{ns}t")
                    values.append("" if text is None else text.text)
                elif value is None:
                    values.append("")
                elif kind == "s":
                    values.append(shared[int(value.text)])
                else:
                    values.append(float(value.text))
            rows.append(values)
    expected = ["Time_s", "q1_rad", "q2_rad", "q3_rad", "q4_rad", "q5_rad", "q6_rad"]
    if rows[0] != expected:
        raise ValueError(f"Unexpected Excel columns: {rows[0]}")
    values = np.asarray(rows[1:], dtype=float)
    return values[:, 0], values[:, 1:7]


def pose(model: mujoco.MjModel, data: mujoco.MjData, site_id: int, q: np.ndarray):
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    position = data.site_xpos[site_id].copy()
    direction = data.site_xmat[site_id].reshape(3, 3)[:, 2].copy()
    return position, direction / np.linalg.norm(direction)


def add_sphere(scene, position, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.full(3, radius),
        position,
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=float),
    )
    scene.ngeom += 1


def add_arrow(scene, start, end, width, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_connector(
        geom, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(start), np.asarray(end)
    )
    geom.rgba[:] = rgba
    scene.ngeom += 1


def update_highlights(viewer, trail, ee_position, ee_direction):
    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0
        add_sphere(scene, trail[0], 0.027, [0.1, 1.0, 0.2, 0.95])
        add_sphere(scene, GOAL_POSITION, 0.035, [1.0, 0.85, 0.0, 0.95])
        add_arrow(
            scene,
            GOAL_POSITION,
            GOAL_POSITION + 0.22 * GOAL_DIRECTION,
            0.012,
            [1.0, 0.1, 0.05, 1.0],
        )
        add_sphere(scene, ee_position, 0.014, [0.0, 0.95, 1.0, 0.95])
        add_arrow(
            scene,
            ee_position,
            ee_position + 0.18 * ee_direction,
            0.009,
            [0.0, 0.95, 1.0, 1.0],
        )
        for point in trail[::5]:
            add_sphere(scene, point, 0.006, [0.15, 0.45, 1.0, 0.55])


def errors(position, direction):
    position_error = np.linalg.norm(GOAL_POSITION - position)
    cosine = np.clip(np.dot(direction, GOAL_DIRECTION), -1.0, 1.0)
    angle_error = math.degrees(math.acos(cosine))
    return position_error, angle_error


def open_control_panel(state):
    """Open a small always-on-top controller for replaying the trajectory."""
    root = tk.Tk()
    root.title("UR5 SAC Controls")
    root.geometry("300x160+30+80")
    root.attributes("-topmost", True)
    root.resizable(False, False)

    status = tk.StringVar(value="Playback is starting...")

    def start():
        state["paused"] = False
        state["restart"] = True

    def pause_resume():
        state["paused"] = not state["paused"]

    def close():
        state["close"] = True

    tk.Label(root, text="UR5 SAC trajectory", font=("Segoe UI", 12, "bold")).pack(pady=(10, 4))
    row = tk.Frame(root)
    row.pack(pady=4)
    tk.Button(row, text="Start", width=10, height=2, command=start).pack(side=tk.LEFT, padx=4)
    tk.Button(row, text="Pause / Resume", width=14, height=2, command=pause_resume).pack(side=tk.LEFT, padx=4)
    tk.Button(row, text="Close", width=8, height=2, command=close).pack(side=tk.LEFT, padx=4)
    tk.Label(root, textvariable=status, font=("Segoe UI", 9)).pack(pady=6)

    def refresh():
        if state.get("viewer_closed"):
            root.destroy()
            return
        status.set(state.get("status", ""))
        root.after(100, refresh)

    root.protocol("WM_DELETE_WINDOW", close)
    refresh()
    root.mainloop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--check", action="store_true", help="Validate without opening the viewer")
    args = parser.parse_args()
    if args.speed <= 0:
        raise ValueError("--speed must be positive")

    timestamps, joints = read_joint_trajectory(args.xlsx.resolve())
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Time_s must be strictly increasing")
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if site_id < 0:
        raise ValueError("attachment_site was not found in the MuJoCo model")

    trail = np.empty((len(joints), 3))
    directions = np.empty((len(joints), 3))
    for index, q in enumerate(joints):
        trail[index], directions[index] = pose(model, data, site_id, q)
    final_position_error, final_angle_error = errors(trail[-1], directions[-1])
    print(f"Samples: {len(timestamps)}, duration: {timestamps[-1]:.3f} s")
    print(f"Final position error: {100 * final_position_error:.4f} cm")
    print(f"Final direction error: {final_angle_error:.4f} deg")
    if args.check:
        return

    state = {
        "paused": False,
        "restart": False,
        "close": False,
        "viewer_closed": False,
        "status": "Playback is starting...",
    }

    def key_callback(keycode):
        if keycode == 32:  # Space
            state["paused"] = not state["paused"]
        elif keycode in (82, 114):  # R/r
            state["restart"] = True

    threading.Thread(target=open_control_panel, args=(state,), daemon=True).start()

    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.38])
            viewer.cam.distance = 1.75
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -22

            while viewer.is_running() and not state["close"]:
                state["restart"] = False
                for index, (stamp, q) in enumerate(zip(timestamps, joints)):
                    if not viewer.is_running() or state["restart"] or state["close"]:
                        break
                    while state["paused"] and viewer.is_running() and not state["restart"] and not state["close"]:
                        state["status"] = f"Paused at {stamp:.2f} s"
                        viewer.sync()
                        time.sleep(0.03)
                    start = time.perf_counter()
                    position, direction = pose(model, data, site_id, q)
                    data.time = stamp
                    position_error, angle_error = errors(position, direction)
                    reached = position_error <= POSITION_TOLERANCE and angle_error <= ANGLE_TOLERANCE_DEG
                    state["status"] = f"Playing: {stamp:.2f} / {timestamps[-1]:.2f} s"
                    update_highlights(viewer, trail[: index + 1], position, direction)
                    viewer.set_texts(
                        [
                        (
                            mujoco.mjtFontScale.mjFONTSCALE_150,
                            mujoco.mjtGridPos.mjGRID_TOPLEFT,
                            "SAC UR5e trajectory\nTime / sample\nPosition error\nDirection error\nStatus",
                            f"{stamp:6.2f} s / {index + 1:3d}\n{100 * position_error:8.4f} cm\n"
                            f"{angle_error:8.4f} deg\n{'GOAL REACHED' if reached else 'MOVING TO GOAL'}",
                        ),
                        (
                            mujoco.mjtFontScale.mjFONTSCALE_150,
                            mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                            "START position (m)\nGOAL position (m)\nGOAL direction\nLegend",
                            f"[{trail[0, 0]:.3f}, {trail[0, 1]:.3f}, {trail[0, 2]:.3f}]\n"
                            f"[{GOAL_POSITION[0]:.3f}, {GOAL_POSITION[1]:.3f}, {GOAL_POSITION[2]:.3f}]\n"
                            f"[{GOAL_DIRECTION[0]:.3f}, {GOAL_DIRECTION[1]:.3f}, {GOAL_DIRECTION[2]:.3f}]\n"
                            "Green: start | Yellow: goal\nRed: goal direction | Cyan: end-effector\nBlue: path",
                        ),
                        (
                            mujoco.mjtFontScale.mjFONTSCALE_100,
                            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                            "Controls",
                            "SPACE: pause/resume | R: replay",
                        ),
                        ]
                    )
                    viewer.sync()
                    if index + 1 < len(timestamps):
                        delay = (timestamps[index + 1] - stamp) / args.speed
                        time.sleep(max(0.0, delay - (time.perf_counter() - start)))

                if not viewer.is_running() or state["close"]:
                    break
                position, direction = pose(model, data, site_id, joints[-1])
                position_error, angle_error = errors(position, direction)
                reached = position_error <= POSITION_TOLERANCE and angle_error <= ANGLE_TOLERANCE_DEG
                state["status"] = "Complete - press Start to replay"
                update_highlights(viewer, trail, position, direction)
                viewer.set_texts(
                    (
                    mujoco.mjtFontScale.mjFONTSCALE_150,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    "PLAYBACK COMPLETE\nPosition error\nDirection error\nResult",
                    f"{100 * position_error:.4f} cm\n{angle_error:.4f} deg\n"
                    f"{'GOAL REACHED' if reached else 'GOAL NOT REACHED'}\nPress R to replay",
                    )
                )
                while viewer.is_running() and not state["restart"] and not state["close"]:
                    viewer.sync()
                    time.sleep(0.03)
            if state["close"] and viewer.is_running():
                viewer.close()
    finally:
        state["viewer_closed"] = True


if __name__ == "__main__":
    main()
