"""C920 camera capability probe and 30-minute stability recorder.

Windows friendly-name enumeration: ``python -m pip install pygrabber``.

OpenCV does not expose DirectShow property ranges/flags. This tool therefore
never claims that focus/exposure controls are locked based on VideoCapture.set.
When those controls cannot be queried natively it records that limitation and
writes an IAMCameraControl/IAMVideoProcAmp implementation design to the report.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


CONTROL_NAMES = (
    "focus", "autofocus", "exposure", "auto_exposure", "white_balance", "gain"
)
DIRECTSHOW_DESIGN = """## DirectShow native control design

OpenCV VideoCapture does not provide DirectShow `IAMCameraControl::GetRange`
or `IAMVideoProcAmp::GetRange` capability flags. Do not infer support from a
successful `VideoCapture.set()` call. A native Windows adapter should:

1. Bind the selected DirectShow capture filter by its moniker/device path, not
   by assuming camera index equals device enumeration order.
2. Query `IAMCameraControl::GetRange` for `Focus` and `Exposure`, and
   `IAMVideoProcAmp::GetRange` for `WhiteBalance` and `Gain`. Record min/max,
   step, default and `Auto`/`Manual` flags.
3. Query `Get` for current values and flags. Only request a value inside the
   reported range and on the reported step; use the documented manual flag.
4. Call `Set`, then `Get` and compare both value and flags with the request.
   Report `LOCKED_VERIFIED` only when the device supports manual mode and the
   read-back matches. Unsupported, failed, or mismatched controls remain
   `NOT_LOCKED` and block production mode.
5. Keep COM graph/filter lifetime on the owning thread and release interfaces
   deterministically. If DirectShow is unavailable, evaluate a Media
   Foundation adapter separately; do not claim equivalent control support.

Relevant interfaces: `IAMCameraControl` (Focus, Exposure) and
`IAMVideoProcAmp` (White Balance, Gain). Autofocus/auto-exposure are normally
represented by the corresponding property flags, not by portable OpenCV
boolean values. This probe does not implement or simulate COM control.
"""


@dataclass
class FrameStats:
    captured_at_utc: str
    elapsed_s: float
    fps_window: float
    width: int
    height: int
    brightness_mean: float
    brightness_std: float
    sharpness_laplacian_var: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def frame_statistics(frame: Any, cv2_module: Any, elapsed_s: float, fps: float) -> FrameStats:
    height, width = frame.shape[:2]
    gray = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    mean, std = cv2_module.meanStdDev(gray)
    laplacian = cv2_module.Laplacian(gray, cv2_module.CV_64F)
    return FrameStats(
        captured_at_utc=_utc_now(), elapsed_s=round(elapsed_s, 3),
        fps_window=round(fps, 3), width=int(width), height=int(height),
        brightness_mean=round(float(mean[0][0]), 3),
        brightness_std=round(float(std[0][0]), 3),
        sharpness_laplacian_var=round(float(laplacian.var()), 3),
    )


def _backend_name(cap: Any, cv2_module: Any) -> str:
    try:
        return str(cap.getBackendName())
    except Exception:
        return "unknown"


def _opencv_control_observation(cap: Any, cv2_module: Any) -> dict[str, dict[str, Any]]:
    """Read current values, explicitly not treating them as capability ranges."""
    props = {
        "focus": "CAP_PROP_FOCUS", "autofocus": "CAP_PROP_AUTOFOCUS",
        "exposure": "CAP_PROP_EXPOSURE", "auto_exposure": "CAP_PROP_AUTO_EXPOSURE",
        "white_balance": "CAP_PROP_WB_TEMPERATURE", "gain": "CAP_PROP_GAIN",
    }
    results: dict[str, dict[str, Any]] = {}
    for name in CONTROL_NAMES:
        prop = getattr(cv2_module, props[name], None)
        value = None
        if prop is not None:
            try:
                raw = float(cap.get(prop))
                value = raw if math.isfinite(raw) else None
            except Exception as exc:
                logger.debug("OpenCV property %s read failed: %s", props[name], exc)
        results[name] = {
            "opencv_property": props[name],
            "current_value": value,
            "capability_query": "NOT_AVAILABLE_IN_OPENCV",
            "set_attempted": False,
            "status": "UNVERIFIED",
            "note": "OpenCV exposes no range/step/auto-manual capability query; no control write was attempted.",
        }
    return results


def discover_camera_name(index: int) -> str:
    """Best-effort friendly name using optional pygrabber; never guesses C920."""
    try:
        from pygrabber.dshow_graph import FilterGraph  # type: ignore
        names = FilterGraph().get_input_devices()
        if 0 <= index < len(names):
            return str(names[index])
    except Exception as exc:
        logger.debug("DirectShow device enumeration unavailable: %s", exc)
    return "Unavailable (install pygrabber for DirectShow friendly names)"


def open_camera(cv2_module: Any, index: int, backend: str) -> Any:
    api = {"DSHOW": cv2_module.CAP_DSHOW, "MSMF": cv2_module.CAP_MSMF}[backend]
    return cv2_module.VideoCapture(index, api)


def run_probe(args: argparse.Namespace, cv2_module: Any | None = None,
              capture_factory: Callable[[int, str], Any] | None = None) -> dict[str, Any]:
    if cv2_module is None:
        import cv2 as cv2_module  # type: ignore
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    requested = {"width": 1920, "height": 1080, "fps": 30}
    candidates: list[dict[str, Any]] = []
    selected = None
    fallback = None
    for index in range(args.max_index + 1):
        for backend in args.backends:
            cap = capture_factory(index, backend) if capture_factory else open_camera(cv2_module, index, backend)
            if not cap or not cap.isOpened():
                try:
                    cap.release()
                except Exception as exc:
                    logger.debug("Release failed for unopened capture: %s", exc)
                continue
            # Resolution/FPS are stream negotiation requests, not camera controls.
            cap.set(cv2_module.CAP_PROP_FRAME_WIDTH, requested["width"])
            cap.set(cv2_module.CAP_PROP_FRAME_HEIGHT, requested["height"])
            cap.set(cv2_module.CAP_PROP_FPS, requested["fps"])
            # pygrabber enumerates DirectShow devices; do not map that order to
            # an MSMF index, which may refer to a different physical camera.
            name = discover_camera_name(index) if backend == "DSHOW" else "Unmapped MSMF device name"
            candidate = {
                "device_index": index, "name": name,
                "backend": _backend_name(cap, cv2_module), "requested": requested,
            }
            candidates.append(candidate)
            if fallback is None:
                fallback = (cap, candidate)
            if "logitech" in name.lower() and "c920" in name.lower():
                selected = (cap, candidate)
                break
            # Continue enumeration: the first open camera is not necessarily the C920.
            if cap is not fallback[0]:
                cap.release()
        if selected is not None:
            break
    if selected is None:
        selected = fallback
    if selected is not None and fallback is not None and selected[0] is not fallback[0]:
        fallback[0].release()
    if selected is None:
        result = {
            "generated_at_utc": _utc_now(), "status": "NO_CAMERA",
            "camera": None, "candidates": candidates,
            "message": "No camera could be opened. No device properties were changed.",
            "directshow_control_design": DIRECTSHOW_DESIGN,
        }
        write_reports(output, result)
        return result

    cap, camera = selected
    camera["friendly_name_source"] = "pygrabber DirectShow enumeration" if "Unavailable" not in camera["name"] else "unavailable"
    if camera["backend"] != "DSHOW":
        camera["friendly_name_source"] = "unmapped for this backend"
    camera["controls"] = _opencv_control_observation(cap, cv2_module)
    camera["controls_status"] = "NOT_LOCKED: OpenCV cannot verify capability ranges and auto/manual flags."
    camera["resolution_fps_request"] = {**requested, "set_return_values_not_used_as_proof": True}

    rows: list[FrameStats] = []
    preview_path = output / "camera_probe_preview.jpg"
    start = time.monotonic()
    last_sample = start
    count = 0
    window_frames = 0
    window_start = start
    csv_path = output / "camera_stability.csv"
    deadline = start + max(0.0, float(args.duration_s))
    try:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(asdict(FrameStats("", 0, 0, 0, 0, 0, 0, 0)).keys()))
            writer.writeheader()
            while time.monotonic() < deadline:
                ok, frame = cap.read()
                now = time.monotonic()
                if not ok or frame is None:
                    camera["read_error"] = "Frame read failed during probe."
                    break
                count += 1
                window_frames += 1
                elapsed_window = now - window_start
                if elapsed_window >= args.sample_interval_s or count == 1:
                    fps = window_frames / elapsed_window if elapsed_window > 0 else 0.0
                    stats = frame_statistics(frame, cv2_module, now - start, fps)
                    rows.append(stats)
                    writer.writerow(asdict(stats))
                    fh.flush()
                    window_frames = 0
                    window_start = now
                if count == 1:
                    cv2_module.imwrite(str(preview_path), frame)
                if args.show:
                    cv2_module.imshow("Camera Probe - press q to stop", frame)
                    if cv2_module.waitKey(1) & 0xFF == ord("q"):
                        break
                last_sample = now
    finally:
        cap.release()
        if args.show:
            cv2_module.destroyAllWindows()
    duration = max(0.0, last_sample - start)
    camera["actual_width"] = rows[0].width if rows else None
    camera["actual_height"] = rows[0].height if rows else None
    camera["measured_fps_overall"] = round(count / duration, 3) if duration else 0.0
    camera["frames_read"] = count
    result = {
        "generated_at_utc": _utc_now(), "status": "COMPLETED" if count else "READ_FAILED",
        "camera": camera, "candidates": candidates, "requested": requested,
        "duration_s_requested": args.duration_s, "duration_s_observed": round(duration, 3),
        "preview_image": str(preview_path) if count else None,
        "stability_csv": str(csv_path),
        "summary": summarize(rows),
        "directshow_control_design": DIRECTSHOW_DESIGN,
        "notes": ["Device friendly-name enumeration requires optional pygrabber.",
                  "OpenCV control properties are observed only; no unverified write was attempted."],
    }
    write_reports(output, result)
    return result


def summarize(rows: list[FrameStats]) -> dict[str, Any]:
    if not rows:
        return {"samples": 0}
    return {
        "samples": len(rows),
        "fps_mean": round(statistics.mean(r.fps_window for r in rows), 3),
        "brightness_mean": round(statistics.mean(r.brightness_mean for r in rows), 3),
        "brightness_std_mean": round(statistics.mean(r.brightness_std for r in rows), 3),
        "sharpness_laplacian_var_mean": round(statistics.mean(r.sharpness_laplacian_var for r in rows), 3),
    }


def write_reports(output: Path, result: dict[str, Any]) -> None:
    json_path = output / "camera_probe_report.json"
    md_path = output / "camera_probe_report.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    camera = result.get("camera") or {}
    lines = [
        "# Camera probe report", "",
        f"- Generated (UTC): {result.get('generated_at_utc')}",
        f"- Status: {result.get('status')}",
        f"- Device: {camera.get('name', 'not detected')}",
        f"- Index / backend: {camera.get('device_index', '—')} / {camera.get('backend', '—')}",
        "- Requested mode: 1920×1080 @ 30 FPS",
        f"- Actual frame: {camera.get('actual_width', '—')}×{camera.get('actual_height', '—')}",
        f"- Measured overall FPS: {camera.get('measured_fps_overall', '—')}",
        f"- Controls: {camera.get('controls_status', 'not queried')}",
        f"- Stability CSV: {result.get('stability_csv', 'not generated')}",
        f"- Preview: {result.get('preview_image', 'not generated')}", "",
        "## Stability summary", "", "```json",
        json.dumps(result.get("summary", {}), ensure_ascii=False, indent=2), "```", "",
        "## Control limitations", "",
        "OpenCV capability ranges/auto-manual flags were unavailable. No focus, autofocus, exposure, auto-exposure, white-balance or gain write was issued, and no control is reported locked.", "",
        result.get("directshow_control_design", DIRECTSHOW_DESIGN),
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe Logitech C920 and record camera stability.")
    parser.add_argument("--output-dir", default="camera_probe_output")
    parser.add_argument("--max-index", type=int, default=8)
    parser.add_argument("--backends", nargs="+", choices=("DSHOW", "MSMF"), default=("DSHOW", "MSMF"))
    parser.add_argument("--duration-s", type=float, default=1800.0, help="Stability recording duration (default 30 min).")
    parser.add_argument("--sample-interval-s", type=float, default=1.0)
    parser.add_argument("--show", action="store_true", help="Show live preview; q ends early.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import cv2  # type: ignore
    except ImportError:
        print("OpenCV is required. Install opencv-python first.", file=sys.stderr)
        return 2
    result = run_probe(args, cv2_module=cv2)
    print(json.dumps({k: result.get(k) for k in ("status", "camera", "stability_csv")}, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
