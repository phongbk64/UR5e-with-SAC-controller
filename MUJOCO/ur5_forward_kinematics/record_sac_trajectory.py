"""Render the MATLAB SAC joint trajectory to an MP4 video with goal highlights."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from run_sac_trajectory import (
    ANGLE_TOLERANCE_DEG,
    DEFAULT_XLSX,
    DEFAULT_XML,
    GOAL_DIRECTION,
    GOAL_POSITION,
    POSITION_TOLERANCE,
    add_arrow,
    add_sphere,
    errors,
    pose,
    read_joint_trajectory,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "ur5_sac_trajectory.mp4"


def load_font(size: int, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def overlay_text(frame, stamp, duration, sample, total, position_error, angle_error):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = load_font(27, bold=True)
    body_font = load_font(21)
    small_font = load_font(18)
    reached = position_error <= POSITION_TOLERANCE and angle_error <= ANGLE_TOLERANCE_DEG

    draw.rounded_rectangle((22, 20, 485, 215), radius=12, fill=(8, 22, 38, 205))
    draw.text((40, 35), "SAC UR5e trajectory", font=title_font, fill=(255, 255, 255, 255))
    draw.text((40, 78), f"Time: {stamp:6.2f} / {duration:.2f} s", font=body_font, fill="white")
    draw.text((40, 110), f"Frame: {sample:4d} / {total}", font=body_font, fill="white")
    draw.text((40, 142), f"Position error: {100 * position_error:7.4f} cm", font=body_font, fill="white")
    draw.text((40, 174), f"Direction error: {angle_error:7.4f} deg", font=body_font, fill="white")

    status = "GOAL REACHED" if reached else "MOVING TO GOAL"
    status_color = (50, 220, 100, 235) if reached else (255, 205, 40, 235)
    draw.rounded_rectangle((22, image.height - 74, 264, image.height - 22), radius=10, fill=(8, 22, 38, 205))
    draw.text((40, image.height - 62), status, font=body_font, fill=status_color)

    legend = "Green: start   Yellow: goal   Red: goal direction\nCyan: end effector   Blue: path"
    box_left = image.width - 455
    draw.rounded_rectangle((box_left, 20, image.width - 22, 98), radius=10, fill=(8, 22, 38, 190))
    draw.multiline_text((box_left + 18, 32), legend, font=small_font, fill="white", spacing=5)
    return np.asarray(image)


def interpolate_joints(timestamps, joints, video_times):
    return np.column_stack(
        [np.interp(video_times, timestamps, joints[:, joint]) for joint in range(6)]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--hold", type=float, default=2.0, help="Seconds to hold the final frame")
    args = parser.parse_args()

    if args.fps <= 0 or args.width <= 0 or args.height <= 0 or args.hold < 0:
        raise ValueError("fps, width and height must be positive; hold must be nonnegative")
    if args.output.suffix.lower() != ".mp4":
        raise ValueError("--output must end in .mp4")

    timestamps, joints = read_joint_trajectory(args.xlsx.resolve())
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Time_s must be strictly increasing")

    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if site_id < 0:
        raise ValueError("attachment_site was not found in the MuJoCo model")

    source_trail = np.empty((len(joints), 3))
    for index, q in enumerate(joints):
        source_trail[index], _ = pose(model, data, site_id, q)

    duration = float(timestamps[-1])
    video_times = np.arange(0.0, duration + 0.5 / args.fps, 1.0 / args.fps)
    video_joints = interpolate_joints(timestamps, joints, video_times)
    total_frames = len(video_times) + int(round(args.hold * args.fps))

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = np.array([0.0, -0.25, 0.38])
    camera.distance = 1.75
    camera.azimuth = 135
    camera.elevation = -22

    args.output.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(
        model,
        height=args.height,
        width=args.width,
        max_geom=10000,
        font_scale=mujoco.mjtFontScale.mjFONTSCALE_150,
    )
    writer = imageio.get_writer(
        str(args.output.resolve()),
        fps=args.fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    )

    last_frame = None
    try:
        for index, (stamp, q) in enumerate(zip(video_times, video_joints), start=1):
            position, direction = pose(model, data, site_id, q)
            data.time = stamp
            renderer.update_scene(data, camera=camera)
            scene = renderer.scene

            add_sphere(scene, source_trail[0], 0.027, [0.1, 1.0, 0.2, 0.95])
            add_sphere(scene, GOAL_POSITION, 0.035, [1.0, 0.85, 0.0, 0.95])
            add_arrow(scene, GOAL_POSITION, GOAL_POSITION + 0.22 * GOAL_DIRECTION, 0.012, [1.0, 0.1, 0.05, 1.0])
            add_sphere(scene, position, 0.014, [0.0, 0.95, 1.0, 0.95])
            add_arrow(scene, position, position + 0.18 * direction, 0.009, [0.0, 0.95, 1.0, 1.0])

            visible = np.searchsorted(timestamps, stamp, side="right")
            for point in source_trail[:visible:5]:
                add_sphere(scene, point, 0.006, [0.15, 0.45, 1.0, 0.55])

            position_error, angle_error = errors(position, direction)
            frame = renderer.render()
            last_frame = overlay_text(
                frame, stamp, duration, index, total_frames, position_error, angle_error
            )
            writer.append_data(last_frame)
            if index == 1 or index % args.fps == 0 or index == len(video_times):
                print(f"Rendering {index}/{len(video_times)} frames")

        for _ in range(int(round(args.hold * args.fps))):
            writer.append_data(last_frame)
    finally:
        writer.close()
        renderer.close()

    final_position, final_direction = pose(model, data, site_id, joints[-1])
    final_position_error, final_angle_error = errors(final_position, final_direction)
    print(f"Video saved: {args.output.resolve()}")
    print(f"Frames: {total_frames}, fps: {args.fps}, duration: {total_frames / args.fps:.2f} s")
    print(f"Final position error: {100 * final_position_error:.4f} cm")
    print(f"Final direction error: {final_angle_error:.4f} deg")


if __name__ == "__main__":
    main()
