"""Dataset sample collection, group-isolation auditing, and real PyTorch MobileNetV3-Small -> ONNX training manager."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import random
import shutil
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CLASSES = ["complete", "crossed_out", "covered", "absent"]
CHALLENGE = "unknown_challenge"
ALL_LABELS = [*CLASSES, CHALLENGE]
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
SAVED_IMAGE_WIDTH = 224
SAVED_IMAGE_HEIGHT = 144
UNGROUPED_TUBE_ID = "UNGROUPED"
UNGROUPED_BATCH_ID = "BATCH_UNGROUPED"
UNGROUPED_VIDEO_GROUP_ID = "VID_UNGROUPED"
ONNX_PARITY_MAX_ABS_DIFF = 1e-3
MANIFEST_FIELDS = [
    "sample_id",
    "path",
    "label",
    "physical_tube_id",
    "batch_id",
    "video_group_id",
    "pipe_color",
    "box_xyxy",
    "crop_box_xyxy",
    "source_digest",
    "captured_at_utc",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_ungrouped_tube(tube_id: str | None) -> bool:
    if not tube_id or not isinstance(tube_id, str):
        return True
    cleaned = tube_id.strip().upper()
    return not cleaned or cleaned == UNGROUPED_TUBE_ID or cleaned.startswith(f"{UNGROUPED_TUBE_ID}_")


def _is_valid_box_coords(box: Sequence[float] | None) -> bool:
    if not box or len(box) != 4:
        return False
    try:
        vals = [float(v) for v in box]
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(v) for v in vals):
        return False
    x1, y1, x2, y2 = vals
    return 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0


def _normalize_crop_box(box_xyxy: Sequence[float] | None) -> list[float]:
    """Validate normalized [x1, y1, x2, y2] in crop coordinates; fall back to full crop [0, 0, 1, 1]."""
    if _is_valid_box_coords(box_xyxy):
        assert_vals = [float(v) for v in box_xyxy]  # type: ignore[union-attr]
        return [
            max(0.0, min(1.0, assert_vals[0])),
            max(0.0, min(1.0, assert_vals[1])),
            max(0.0, min(1.0, assert_vals[2])),
            max(0.0, min(1.0, assert_vals[3])),
        ]
    return [0.0, 0.0, 1.0, 1.0]


def _frame_box_to_crop_box(
    box_xyxy: Sequence[float] | None,
    crop_bounds_xyxy: Sequence[float] | None = None,
) -> list[float]:
    """Convert full-camera-frame normalized box into normalized saved-crop coordinates [0, 1]."""
    if not _is_valid_box_coords(box_xyxy):
        return [0.0, 0.0, 1.0, 1.0]
    bx1, by1, bx2, by2 = (float(v) for v in box_xyxy)  # type: ignore[union-attr]
    bounds = crop_bounds_xyxy if _is_valid_box_coords(crop_bounds_xyxy) else (bx1, by1, bx2, by2)
    cx1, cy1, cx2, cy2 = (float(v) for v in bounds)  # type: ignore[union-attr]
    cw = cx2 - cx1
    ch = cy2 - cy1
    if cw <= 0.0 or ch <= 0.0:
        return [0.0, 0.0, 1.0, 1.0]
    nx1 = max(0.0, min(1.0, (bx1 - cx1) / cw))
    ny1 = max(0.0, min(1.0, (by1 - cy1) / ch))
    nx2 = max(0.0, min(1.0, (bx2 - cx1) / cw))
    ny2 = max(0.0, min(1.0, (by2 - cy1) / ch))
    if nx2 <= nx1 or ny2 <= ny1:
        return [0.0, 0.0, 1.0, 1.0]
    return [nx1, ny1, nx2, ny2]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex[:6]}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError as cleanup_exc:
            logger.debug("Failed to remove temp JSON file %s: %s", tmp_path, cleanup_exc)
        raise


class TrainingManager:
    """Manages shop-floor ROI dataset collection, grouped split auditing, and live PyTorch -> ONNX training."""

    def __init__(self, data_root_getter: Callable[[], Path]) -> None:
        self._data_root_getter = data_root_getter
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._pending_metrics_write: tuple[Path, dict[str, Any]] | None = None
        self._status: dict[str, Any] = {
            "state": "IDLE",
            "message": "等待启动训练",
            "current_epoch": 0,
            "total_epochs": 0,
            "progress_pct": 0,
            "history": [],
            "latest_run": None,
            "exported": False,
            "activated": False,
            "activation_status": None,
            "metrics_persisted": True,
            "metrics_write_error": None,
            "error": None,
            "updated_at_utc": _utc_now(),
        }

    def _dataset_dir(self) -> Path:
        d = self._data_root_getter() / "dataset"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _manifest_path(self) -> Path:
        return self._dataset_dir() / "samples_manifest.csv"

    def _runs_dir(self) -> Path:
        d = self._data_root_getter() / "runs" / "label_classifier"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _read_manifest_rows(self) -> list[dict[str, str]]:
        path = self._manifest_path()
        if not path.is_file():
            return []
        ds_root = self._dataset_dir().resolve()
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = []
            for r in reader:
                raw_path = (r.get("path") or "").strip()
                if not raw_path:
                    continue
                img_path = Path(raw_path)
                if not img_path.is_absolute():
                    img_path = (path.parent / img_path).resolve()
                else:
                    img_path = img_path.resolve()
                if ds_root not in img_path.parents:
                    logger.warning("Skipping manifest entry outside dataset directory: %s", raw_path)
                    continue
                if img_path.is_file():
                    rows.append({**r, "abs_path": str(img_path)})
            return rows

    def _append_manifest_row(self, row: dict[str, str]) -> None:
        path = self._manifest_path()
        if path.is_file():
            with path.open("r", newline="", encoding="utf-8-sig") as rf:
                reader = csv.DictReader(rf)
                existing_fields = list(reader.fieldnames or [])
                if existing_fields != MANIFEST_FIELDS:
                    existing_rows = list(reader)
                    existing_rows.append(row)
                    self._write_all_manifest_rows(existing_rows)
                    return
            with path.open("a", newline="", encoding="utf-8-sig") as af:
                writer = csv.DictWriter(af, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
                writer.writerow(row)
            return
        with path.open("w", newline="", encoding="utf-8-sig") as wf:
            writer = csv.DictWriter(wf, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(row)

    def _try_unlink_in_dataset(self, file_path: Path) -> tuple[bool, bool, str | None]:
        """Attempt to unlink a dataset sample file.

        Returns `(can_remove_manifest_row, file_was_unlinked, error_message)`.
        Non-ENOENT OSError returns `(False, False, err)` so the caller retains the manifest row for retry.
        """
        ds_root = self._dataset_dir().resolve()
        try:
            resolved = file_path.resolve()
        except OSError as exc:
            logger.warning("Failed to resolve dataset sample path %s: %s", file_path, exc)
            return False, False, str(exc)
        if ds_root not in resolved.parents:
            logger.warning("Refused to delete file outside dataset directory: %s", resolved)
            return False, False, "file path outside dataset directory"
        try:
            resolved.unlink()
            return True, True, None
        except FileNotFoundError:
            return True, False, None
        except OSError as exc:
            logger.warning("Failed to remove dataset sample file %s: %s", file_path, exc)
            return False, False, str(exc)

    def _safe_unlink_in_dataset(self, file_path: Path) -> bool:
        _, unlinked, _ = self._try_unlink_in_dataset(file_path)
        return unlinked

    @staticmethod
    def _empty_coco_doc() -> dict[str, Any]:
        return {
            "images": [],
            "annotations": [],
            "categories": [{"id": i + 1, "name": name} for i, name in enumerate(ALL_LABELS)],
        }

    def _load_coco_doc(self, coco_path: Path) -> tuple[dict[str, Any] | None, bool]:
        """Load existing COCO JSON strictly; never treat read/decode/parse failures as an empty dataset."""
        if not coco_path.is_file():
            return self._empty_coco_doc(), True
        try:
            raw = coco_path.read_text(encoding="utf-8")
            doc = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.error("Preserving existing COCO annotations file after read failure (%s): %s", coco_path, exc)
            return None, False
        if not isinstance(doc, dict) or not isinstance(doc.get("images"), list) or not isinstance(doc.get("annotations"), list):
            logger.error("Preserving malformed COCO annotations file without overwriting: %s", coco_path)
            return None, False
        if not isinstance(doc.get("categories"), list) or not doc["categories"]:
            doc["categories"] = [{"id": i + 1, "name": name} for i, name in enumerate(ALL_LABELS)]
        return doc, True

    @staticmethod
    def _append_coco_entry(
        coco: dict[str, Any],
        img_id: int,
        ann_id: int,
        sample_id: str,
        rel_path: str,
        label: str,
        crop_box_xyxy: Sequence[float] | None,
        image_width: int = SAVED_IMAGE_WIDTH,
        image_height: int = SAVED_IMAGE_HEIGHT,
    ) -> None:
        width = max(1, int(image_width))
        height = max(1, int(image_height))
        b = _normalize_crop_box(crop_box_xyxy)
        x1 = max(0.0, min(float(width), b[0] * width))
        y1 = max(0.0, min(float(height), b[1] * height))
        x2 = max(0.0, min(float(width), b[2] * width))
        y2 = max(0.0, min(float(height), b[3] * height))
        bw = max(1.0, min(float(width) - x1, x2 - x1))
        bh = max(1.0, min(float(height) - y1, y2 - y1))
        coco["images"].append(
            {
                "id": img_id,
                "file_name": rel_path,
                "sample_id": sample_id,
                "width": width,
                "height": height,
            }
        )
        coco["annotations"].append(
            {
                "id": ann_id,
                "image_id": img_id,
                "category_id": ALL_LABELS.index(label) + 1 if label in ALL_LABELS else 1,
                "bbox": [round(x1, 1), round(y1, 1), round(bw, 1), round(bh, 1)],
                "area": round(bw * bh, 1),
                "iscrowd": 0,
            }
        )

    def _append_coco_annotations(self, entries: Sequence[dict[str, Any]]) -> bool:
        if not entries:
            return True
        coco_path = self._dataset_dir() / "annotations_coco.json"
        coco, ok = self._load_coco_doc(coco_path)
        if not ok or coco is None:
            return False
        existing_img_ids = [int(im.get("id", 0)) for im in coco.get("images", []) if isinstance(im, dict)]
        existing_ann_ids = [int(an.get("id", 0)) for an in coco.get("annotations", []) if isinstance(an, dict)]
        next_img_id = (max(existing_img_ids) + 1) if existing_img_ids else 1
        next_ann_id = (max(existing_ann_ids) + 1) if existing_ann_ids else 1
        for entry in entries:
            self._append_coco_entry(
                coco=coco,
                img_id=next_img_id,
                ann_id=next_ann_id,
                sample_id=str(entry["sample_id"]),
                rel_path=str(entry["rel_path"]),
                label=str(entry["label"]),
                crop_box_xyxy=entry.get("crop_box_xyxy"),
                image_width=int(entry.get("image_width") or SAVED_IMAGE_WIDTH),
                image_height=int(entry.get("image_height") or SAVED_IMAGE_HEIGHT),
            )
            next_img_id += 1
            next_ann_id += 1
        try:
            _write_json_atomic(coco_path, coco)
        except OSError as exc:
            logger.error("Failed to persist COCO annotations (%s); preserving existing file: %s", coco_path, exc)
            return False
        return True

    def _append_coco_annotation(
        self,
        sample_id: str,
        rel_path: str,
        label: str,
        box_xyxy: Sequence[float] | None,
        *,
        image_width: int = SAVED_IMAGE_WIDTH,
        image_height: int = SAVED_IMAGE_HEIGHT,
    ) -> bool:
        return self._append_coco_annotations(
            [
                {
                    "sample_id": sample_id,
                    "rel_path": rel_path,
                    "label": label,
                    "crop_box_xyxy": box_xyxy,
                    "image_width": image_width,
                    "image_height": image_height,
                }
            ]
        )

    @staticmethod
    def _parse_csv_box(raw_box: str | None) -> list[float] | None:
        text = (raw_box or "").strip()
        if not text:
            return None
        parts = text.split(",")
        if len(parts) != 4:
            return None
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            return None
        return vals if _is_valid_box_coords(vals) else None

    def rebuild_coco_from_manifest(self) -> dict[str, Any]:
        """Rebuild annotations_coco.json cleanly from the authoritative manifest CSV and saved image dimensions."""
        with self._lock:
            rows = self._read_manifest_rows()
            coco = self._empty_coco_doc()
            for idx, r in enumerate(rows, start=1):
                abs_path = Path(r["abs_path"])
                img = cv2.imread(str(abs_path), cv2.IMREAD_COLOR)
                if img is not None and img.size > 0:
                    img_h, img_w = int(img.shape[0]), int(img.shape[1])
                else:
                    img_w, img_h = SAVED_IMAGE_WIDTH, SAVED_IMAGE_HEIGHT
                sid = str(r.get("sample_id") or "")
                # Prefer persisted crop-space coordinates; keep legacy fallback for older rows without crop_box_xyxy.
                crop_box = self._parse_csv_box(r.get("crop_box_xyxy"))
                if crop_box is None and sid.startswith("STD_"):
                    crop_box = self._parse_csv_box(r.get("box_xyxy"))
                self._append_coco_entry(
                    coco=coco,
                    img_id=idx,
                    ann_id=idx,
                    sample_id=sid,
                    rel_path=str(r.get("path") or ""),
                    label=str(r.get("label") or CLASSES[0]),
                    crop_box_xyxy=crop_box,
                    image_width=img_w,
                    image_height=img_h,
                )
            coco_path = self._dataset_dir() / "annotations_coco.json"
            _write_json_atomic(coco_path, coco)
        return {"rebuilt_annotations": len(coco["annotations"]), "coco_path": str(coco_path)}

    def _sync_coco_delete(self, deleted_sample_ids: set[str]) -> None:
        coco_path = self._dataset_dir() / "annotations_coco.json"
        if not deleted_sample_ids or not coco_path.is_file():
            return
        coco, ok = self._load_coco_doc(coco_path)
        if not ok or coco is None:
            return
        removed_img_ids = {
            im.get("id") for im in coco.get("images", []) if isinstance(im, dict) and im.get("sample_id") in deleted_sample_ids
        }
        if not removed_img_ids:
            return
        coco["images"] = [im for im in coco.get("images", []) if not (isinstance(im, dict) and im.get("id") in removed_img_ids)]
        coco["annotations"] = [an for an in coco.get("annotations", []) if not (isinstance(an, dict) and an.get("image_id") in removed_img_ids)]
        try:
            _write_json_atomic(coco_path, coco)
        except OSError as exc:
            logger.warning("Failed to persist COCO sync delete (%s): %s", coco_path, exc)

    def _sync_coco_update(self, sample_id: str, new_rel_path: str, new_label: str) -> None:
        coco_path = self._dataset_dir() / "annotations_coco.json"
        if not coco_path.is_file():
            return
        coco, ok = self._load_coco_doc(coco_path)
        if not ok or coco is None:
            return
        updated_img_ids: set[Any] = set()
        for im in coco.get("images", []):
            if isinstance(im, dict) and im.get("sample_id") == sample_id:
                im["file_name"] = new_rel_path
                updated_img_ids.add(im.get("id"))
        if not updated_img_ids:
            return
        cat_id = ALL_LABELS.index(new_label) + 1 if new_label in ALL_LABELS else 1
        for an in coco.get("annotations", []):
            if isinstance(an, dict) and an.get("image_id") in updated_img_ids:
                an["category_id"] = cat_id
        try:
            _write_json_atomic(coco_path, coco)
        except OSError as exc:
            logger.warning("Failed to persist COCO sync update (%s): %s", coco_path, exc)

    def _save_sample_image(
        self,
        crop_bgr: np.ndarray,
        label: str,
        pipe_color: str,
        physical_tube_id: str | None,
        batch_id: str | None,
        box_xyxy: Sequence[float] | None,
        id_prefix: str = "SMP",
        box_fallback: str = "auto_tracked",
        coco_box_xyxy: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        if label not in ALL_LABELS:
            raise ValueError(f"Unsupported sample label: {label}")
        if pipe_color not in {"yellow", "blue"}:
            raise ValueError(f"Unsupported pipe_color: {pipe_color}")
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("Empty ROI crop image")

        ds_dir = self._dataset_dir()
        img_dir = ds_dir / "images" / label
        img_dir.mkdir(parents=True, exist_ok=True)

        now_str = datetime.now().strftime("%H%M%S")
        sample_id = f"{id_prefix}_{label[:4].upper()}_{now_str}_{uuid.uuid4().hex[:5].upper()}"
        rel_path = Path("images") / label / f"{sample_id}.jpg"
        abs_path = ds_dir / rel_path

        resized = cv2.resize(crop_bgr, (SAVED_IMAGE_WIDTH, SAVED_IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
        saved_h, saved_w = int(resized.shape[0]), int(resized.shape[1])
        ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        if not ok:
            raise RuntimeError("Failed to encode ROI sample JPEG")
        abs_path.write_bytes(bytes(buf))
        sha = _sha256_file(abs_path)

        box_str = ",".join(f"{float(v):.4f}" for v in box_xyxy) if (box_xyxy and len(box_xyxy) == 4) else box_fallback
        resolved_coco_box = _normalize_crop_box(coco_box_xyxy)

        clean_tube = physical_tube_id.strip()[:64] if isinstance(physical_tube_id, str) and physical_tube_id.strip() else None
        clean_batch = batch_id.strip()[:64] if isinstance(batch_id, str) and batch_id.strip() else None

        with self._lock:
            # Never assign fake rotating independent tube IDs to ungrouped captures;
            # keep ungrouped captures together in a single non-independent bucket.
            if clean_tube and not _is_ungrouped_tube(clean_tube):
                tube_id = clean_tube
                b_id = clean_batch or UNGROUPED_BATCH_ID
                v_id = f"VID_{clean_tube}"
            else:
                tube_id = UNGROUPED_TUBE_ID
                b_id = clean_batch or UNGROUPED_BATCH_ID
                v_id = UNGROUPED_VIDEO_GROUP_ID

            row = {
                "sample_id": sample_id,
                "path": rel_path.as_posix(),
                "label": label,
                "physical_tube_id": tube_id,
                "batch_id": b_id,
                "video_group_id": v_id,
                "pipe_color": pipe_color,
                "box_xyxy": box_str,
                "crop_box_xyxy": ",".join(f"{v:.4f}" for v in resolved_coco_box),
                "source_digest": "",
                "captured_at_utc": _utc_now(),
            }
            self._append_manifest_row(row)
            coco_synced = self._append_coco_annotation(
                sample_id,
                rel_path.as_posix(),
                label,
                resolved_coco_box,
                image_width=saved_w,
                image_height=saved_h,
            )

        return {
            **row,
            "sha256": sha,
            "coco_synced": coco_synced,
            "image_url": f"/api/dataset/sample/{sample_id}/image",
        }

    def capture_sample(
        self,
        crop_bgr: np.ndarray,
        label: str,
        pipe_color: str = "yellow",
        physical_tube_id: str | None = None,
        batch_id: str | None = None,
        box_xyxy: list[float] | None = None,
        crop_bounds_xyxy: list[float] | None = None,
    ) -> dict[str, Any]:
        # Full-camera-frame custom boxes are converted into the saved crop's coordinate space for COCO.
        coco_crop_box = _frame_box_to_crop_box(box_xyxy, crop_bounds_xyxy=crop_bounds_xyxy)
        return self._save_sample_image(
            crop_bgr=crop_bgr,
            label=label,
            pipe_color=pipe_color,
            physical_tube_id=physical_tube_id,
            batch_id=batch_id,
            box_xyxy=box_xyxy,
            id_prefix="SMP",
            box_fallback="auto_tracked",
            coco_box_xyxy=coco_crop_box,
        )

    def upload_sample_bytes(
        self,
        image_bytes: bytes,
        label: str,
        pipe_color: str = "yellow",
        physical_tube_id: str | None = None,
        batch_id: str | None = None,
    ) -> dict[str, Any]:
        """Validate uploaded JPEG/PNG bytes via cv2.imdecode, save into dataset/images/<label>/, and record in manifest & COCO."""
        if label not in ALL_LABELS:
            raise ValueError(f"Unsupported sample label: {label}")
        if pipe_color not in {"yellow", "blue"}:
            raise ValueError(f"Unsupported pipe_color: {pipe_color}")
        if not image_bytes:
            raise ValueError("Empty uploaded image bytes")

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if decoded is None or decoded.size == 0:
            raise ValueError("Invalid or corrupted JPEG/PNG image bytes")

        clean_tube = physical_tube_id.strip()[:64] if physical_tube_id and physical_tube_id.strip() else None
        clean_batch = batch_id.strip()[:64] if batch_id and batch_id.strip() else None
        return self._save_sample_image(
            crop_bgr=decoded,
            label=label,
            pipe_color=pipe_color,
            physical_tube_id=clean_tube,
            batch_id=clean_batch,
            box_xyxy=[0.0, 0.0, 1.0, 1.0],
            id_prefix="UPL",
            box_fallback="uploaded_image",
            coco_box_xyxy=[0.0, 0.0, 1.0, 1.0],
        )

    def list_samples(
        self,
        label: str | None = None,
        query: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Return filtered dataset samples (newest first) with full metadata and image URL."""
        clean_label = label.strip() if isinstance(label, str) else ""
        if clean_label and clean_label != "all" and clean_label not in ALL_LABELS:
            raise ValueError(f"Unsupported sample label filter: {clean_label}")
        clean_query = query.strip().lower() if isinstance(query, str) else ""
        clamped_limit = max(1, min(int(limit), 2000))

        with self._lock:
            rows = self._read_manifest_rows()

        results: list[dict[str, Any]] = []
        for r in reversed(rows):
            if clean_label and clean_label != "all" and r.get("label") != clean_label:
                continue
            if clean_query:
                haystack = " ".join(
                    [
                        r.get("sample_id") or "",
                        r.get("label") or "",
                        r.get("physical_tube_id") or "",
                        r.get("batch_id") or "",
                        r.get("pipe_color") or "",
                    ]
                ).lower()
                if clean_query not in haystack:
                    continue
            sample_id = r.get("sample_id") or ""
            results.append(
                {
                    "sample_id": sample_id,
                    "label": r.get("label") or "",
                    "physical_tube_id": r.get("physical_tube_id") or "",
                    "batch_id": r.get("batch_id") or "",
                    "pipe_color": r.get("pipe_color") or "yellow",
                    "box_xyxy": r.get("box_xyxy") or "auto_tracked",
                    "captured_at_utc": r.get("captured_at_utc") or "",
                    "image_url": f"/api/dataset/sample/{sample_id}/image",
                }
            )
            if len(results) >= clamped_limit:
                break
        return results

    def get_sample_image_path(self, sample_id: str) -> Path | None:
        ds_root = self._dataset_dir().resolve()
        for r in self._read_manifest_rows():
            if r.get("sample_id") == sample_id:
                p = Path(r["abs_path"]).resolve()
                if ds_root in p.parents and p.is_file():
                    return p
        return None

    def _write_all_manifest_rows(self, rows: list[dict[str, str]]) -> None:
        path = self._manifest_path()
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def update_sample_annotation(
        self,
        sample_id: str,
        new_label: str | None = None,
        physical_tube_id: str | None = None,
        batch_id: str | None = None,
        pipe_color: str | None = None,
    ) -> dict[str, Any]:
        """Allow manual re-annotation (label, tube group, batch ID, pipe color) of any captured sample."""
        if new_label is not None and new_label not in ALL_LABELS:
            raise ValueError(f"Unsupported label: {new_label}")
        with self._lock:
            rows = self._read_manifest_rows()
            updated_row: dict[str, str] | None = None
            label_changed = False
            for r in rows:
                if r.get("sample_id") == sample_id:
                    if new_label and new_label != r.get("label"):
                        ds_dir = self._dataset_dir().resolve()
                        old_abs = Path(r["abs_path"]).resolve()
                        new_dir = ds_dir / "images" / new_label
                        new_dir.mkdir(parents=True, exist_ok=True)
                        new_rel = Path("images") / new_label / old_abs.name
                        new_abs = (ds_dir / new_rel).resolve()
                        if (
                            ds_dir in old_abs.parents
                            and ds_dir in new_abs.parents
                            and old_abs.is_file()
                            and old_abs != new_abs
                        ):
                            old_abs.replace(new_abs)
                        r["label"] = new_label
                        r["path"] = new_rel.as_posix()
                        label_changed = True
                    if physical_tube_id:
                        cleaned_tube = physical_tube_id.strip()[:64]
                        if _is_ungrouped_tube(cleaned_tube):
                            r["physical_tube_id"] = UNGROUPED_TUBE_ID
                            r["video_group_id"] = UNGROUPED_VIDEO_GROUP_ID
                        else:
                            r["physical_tube_id"] = cleaned_tube
                            r["video_group_id"] = f"VID_{cleaned_tube}"
                    if batch_id:
                        r["batch_id"] = batch_id.strip()[:64]
                    if pipe_color:
                        r["pipe_color"] = pipe_color.strip()
                    updated_row = dict(r)
                    break
            if updated_row is None:
                raise KeyError(f"Sample not found: {sample_id}")
            self._write_all_manifest_rows(rows)
            if label_changed:
                self._sync_coco_update(sample_id, updated_row["path"], updated_row["label"])
        updated_row["image_url"] = f"/api/dataset/sample/{sample_id}/image"
        return updated_row

    def delete_sample(self, sample_id: str) -> dict[str, Any]:
        """Delete a mislabeled or blurry sample from disk first, retaining manifest row if unlink fails."""
        with self._lock:
            rows = self._read_manifest_rows()
            kept: list[dict[str, str]] = []
            target_row: dict[str, str] | None = None
            for r in rows:
                if r.get("sample_id") == sample_id:
                    target_row = r
                else:
                    kept.append(r)
            if target_row is None:
                raise KeyError(f"Sample not found: {sample_id}")
            can_remove, unlinked, err = self._try_unlink_in_dataset(Path(target_row["abs_path"]))
            if not can_remove:
                raise OSError(f"Failed to delete sample file for {sample_id}: {err}")
            self._write_all_manifest_rows(kept)
            self._sync_coco_delete({sample_id})
        return {"deleted": sample_id, "deleted_file": unlinked, "remaining_samples": len(kept)}

    def batch_delete_samples(
        self,
        sample_ids: list[str] | None = None,
        label: str | None = None,
    ) -> dict[str, Any]:
        """Delete matching samples from disk first; retain manifest rows for any files that fail to unlink."""
        id_set = {s.strip() for s in (sample_ids or []) if isinstance(s, str) and s.strip()}
        clean_label = label.strip() if isinstance(label, str) and label.strip() else None
        if clean_label is not None and clean_label not in ALL_LABELS:
            raise ValueError(f"Unsupported label: {clean_label}")
        if not id_set and clean_label is None:
            raise ValueError("Must specify sample_ids or label to batch delete")

        with self._lock:
            rows = self._read_manifest_rows()
            kept: list[dict[str, str]] = []
            removed_ids: list[str] = []
            failed_deletions: list[dict[str, str]] = []
            deleted_files = 0
            for r in rows:
                sid = r.get("sample_id") or ""
                lbl = r.get("label") or ""
                if id_set and clean_label is not None:
                    should_remove = sid in id_set and lbl == clean_label
                elif id_set:
                    should_remove = sid in id_set
                else:
                    should_remove = lbl == clean_label
                if should_remove:
                    can_remove, unlinked, err = self._try_unlink_in_dataset(Path(r["abs_path"]))
                    if can_remove:
                        removed_ids.append(sid)
                        if unlinked:
                            deleted_files += 1
                    else:
                        kept.append(r)
                        failed_deletions.append(
                            {
                                "sample_id": sid,
                                "path": str(r.get("path") or ""),
                                "error": err or "unlink failed",
                            }
                        )
                else:
                    kept.append(r)
            self._write_all_manifest_rows(kept)
            if removed_ids:
                self._sync_coco_delete(set(removed_ids))
        return {
            "deleted_count": len(removed_ids),
            "deleted_sample_ids": removed_ids,
            "deleted_files": deleted_files,
            "failed_count": len(failed_deletions),
            "failed_deletions": failed_deletions,
            "remaining_samples": len(kept),
        }

    def clear_dataset(self) -> dict[str, Any]:
        """Clear dataset samples from disk first, retaining manifest rows for any files that fail to unlink."""
        with self._lock:
            rows = self._read_manifest_rows()
            kept: list[dict[str, str]] = []
            removed_ids: list[str] = []
            failed_deletions: list[dict[str, str]] = []
            deleted_files = 0
            for r in rows:
                sid = r.get("sample_id") or ""
                can_remove, unlinked, err = self._try_unlink_in_dataset(Path(r["abs_path"]))
                if can_remove:
                    removed_ids.append(sid)
                    if unlinked:
                        deleted_files += 1
                else:
                    kept.append(r)
                    failed_deletions.append(
                        {
                            "sample_id": sid,
                            "path": str(r.get("path") or ""),
                            "error": err or "unlink failed",
                        }
                    )
            self._write_all_manifest_rows(kept)
            coco_path = self._dataset_dir() / "annotations_coco.json"
            if not kept:
                try:
                    coco_path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    logger.warning("Failed to remove COCO annotations file during clear: %s", exc)
                    failed_deletions.append(
                        {
                            "sample_id": "",
                            "path": "annotations_coco.json",
                            "error": str(exc),
                        }
                    )
            elif removed_ids:
                self._sync_coco_delete(set(removed_ids))
            summary = self._build_dataset_summary(kept)
        summary["cleared"] = len(failed_deletions) == 0
        summary["deleted_count"] = len(removed_ids)
        summary["deleted_files"] = deleted_files
        summary["failed_count"] = len(failed_deletions)
        summary["failed_deletions"] = failed_deletions
        return summary

    @staticmethod
    def _dir_usage(directory: Path) -> tuple[int, int]:
        if not directory.is_dir():
            return 0, 0
        file_count = 0
        total_bytes = 0
        for item in directory.rglob("*"):
            try:
                if item.is_file():
                    file_count += 1
                    total_bytes += item.stat().st_size
            except OSError:
                continue
        return file_count, total_bytes

    def storage_stats(self) -> dict[str, Any]:
        """Compute disk usage, file counts, COCO annotations count, and model runs overview."""
        ds_dir = self._dataset_dir()
        runs_dir = self._runs_dir()
        summary = self.get_dataset_summary()
        ds_files, ds_bytes = self._dir_usage(ds_dir)
        run_files, run_bytes = self._dir_usage(runs_dir)
        run_count = sum(1 for item in runs_dir.iterdir() if item.is_dir()) if runs_dir.is_dir() else 0

        coco_annotations = 0
        coco_path = ds_dir / "annotations_coco.json"
        if coco_path.is_file():
            try:
                coco_data = json.loads(coco_path.read_text(encoding="utf-8"))
                anns = coco_data.get("annotations")
                if isinstance(anns, list):
                    coco_annotations = len(anns)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Failed to parse COCO annotations for storage stats: %s", exc)

        return {
            "dataset_dir": str(ds_dir),
            "manifest_path": str(self._manifest_path()),
            "total_samples": summary["total_samples"],
            "class_counts": summary["class_counts"],
            "independent_tube_groups": summary["independent_tube_groups"],
            "coco_annotations": coco_annotations,
            "dataset_files": ds_files,
            "dataset_size_mb": round(ds_bytes / (1024 * 1024), 3),
            "runs_dir": str(runs_dir),
            "run_count": run_count,
            "models_files": run_files,
            "models_size_mb": round(run_bytes / (1024 * 1024), 3),
        }

    @staticmethod
    def _synthesize_roi_sample_with_box(
        base_bgr: np.ndarray | None,
        label: str,
        pipe_color: str,
        variant_idx: int,
        rng: random.Random,
    ) -> tuple[np.ndarray, list[float]]:
        """Create realistic, distinct ROI training images and return (bgr_image, normalized_crop_box_xyxy)."""
        h, w = SAVED_IMAGE_HEIGHT, SAVED_IMAGE_WIDTH
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Base tube background
        if pipe_color == "blue":
            bg_color = (175 + rng.randint(-12, 12), 105 + rng.randint(-10, 10), 35 + rng.randint(-8, 8))
        else:
            bg_color = (42 + rng.randint(-10, 10), 178 + rng.randint(-12, 12), 215 + rng.randint(-12, 12))
        canvas[:, :] = bg_color

        if base_bgr is not None and base_bgr.size > 0:
            resized_base = cv2.resize(base_bgr, (w, h), interpolation=cv2.INTER_AREA)
            canvas = cv2.addWeighted(canvas, 0.55, resized_base, 0.45, 0)

        # Tube wall specular highlight lines
        y_top = 18 + rng.randint(-3, 3)
        y_bot = h - 18 + rng.randint(-3, 3)
        cv2.line(canvas, (0, y_top), (w, y_top), (235, 240, 245), 2, cv2.LINE_AA)
        cv2.line(canvas, (0, y_bot), (w, y_bot), (90, 95, 100), 2, cv2.LINE_AA)

        lx1 = 34 + rng.randint(-8, 8)
        ly1 = 28 + rng.randint(-6, 6)
        lx2 = w - 34 + rng.randint(-8, 8)
        ly2 = h - 28 + rng.randint(-6, 6)
        box_norm = [round(lx1 / w, 4), round(ly1 / h, 4), round(lx2 / w, 4), round(ly2 / h, 4)]

        if label == "complete":
            # Crisp white anti-mistake label with printed serial bars
            cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), (242, 244, 246), -1)
            cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), (180, 185, 190), 1)
            for bx in range(lx1 + 14, lx2 - 16, 9 + (variant_idx % 3)):
                bh = 22 + ((bx + variant_idx) % 14)
                cv2.line(canvas, (bx, ly1 + 18), (bx, ly1 + 18 + bh), (45, 50, 58), 2)
            cv2.line(canvas, (lx1 + 14, ly2 - 18), (lx2 - 18, ly2 - 18), (70, 75, 85), 2)

        elif label == "crossed_out":
            # White label with bold dark marker cross-out strokes (A class)
            cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), (236, 238, 240), -1)
            for bx in range(lx1 + 16, lx2 - 16, 11):
                cv2.line(canvas, (bx, ly1 + 20), (bx, ly1 + 42), (65, 70, 78), 2)
            thickness = 7 + (variant_idx % 4)
            cv2.line(
                canvas,
                (lx1 + 6, ly1 + 8),
                (lx2 - 6, ly2 - 8),
                (18, 20, 24),
                thickness,
                cv2.LINE_AA,
            )
            cv2.line(
                canvas,
                (lx1 + 8, ly2 - 10),
                (lx2 - 8, ly1 + 10),
                (22, 24, 28),
                thickness - 1,
                cv2.LINE_AA,
            )

        elif label == "covered":
            # Opaque dark/orange industrial tape covering the label region (B class)
            cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), (235, 235, 235), -1)
            tx1 = lx1 - 6 + rng.randint(-3, 3)
            ty1 = ly1 - 4 + rng.randint(-3, 3)
            tx2 = lx2 + 6 + rng.randint(-3, 3)
            ty2 = ly2 + 4 + rng.randint(-3, 3)
            tape_color = (38, 72, 165) if variant_idx % 2 == 0 else (48, 52, 58)
            cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2), tape_color, -1)

        elif label == "absent":
            # Label peeled off (C class): only bare tube body + faint adhesive residue
            for rx in range(lx1 + 15, lx2 - 15, 18):
                cv2.circle(canvas, (rx, (ly1 + ly2) // 2), 4, bg_color, -1)

        else:  # unknown_challenge
            # Extreme glare / heavy motion blur challenge sample
            cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), (250, 252, 255), -1)
            cv2.circle(canvas, (w // 2 + rng.randint(-15, 15), h // 2), 48, (255, 255, 255), -1)
            canvas = cv2.GaussianBlur(canvas, (15, 15), 5.0)

        # Add unique sensor grain so every generated image has a distinct SHA256 and realistic noise
        noise = np.random.RandomState(seed=2026 + variant_idx * 97 + ALL_LABELS.index(label) * 13).normal(0, 4.5, canvas.shape)
        noisy = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy, box_norm

    @classmethod
    def _synthesize_roi_sample(
        cls,
        base_bgr: np.ndarray | None,
        label: str,
        pipe_color: str,
        variant_idx: int,
        rng: random.Random,
    ) -> np.ndarray:
        """Create realistic, distinct ROI training images for each class with physical texture and noise."""
        img, _ = cls._synthesize_roi_sample_with_box(base_bgr, label, pipe_color, variant_idx, rng)
        return img

    def bootstrap_standard_dataset(
        self,
        live_crop_bgr: np.ndarray | None = None,
        live_crops: Sequence[np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Generate 48 labeled synthetic ROI samples + 4 challenge samples across 4 groups, writing both manifest and COCO.

        Source-lineage isolation (BE-24 / R2-15): never blends the same live source crop across multiple groups,
        and pins previously seen source digests to their original group across repeated bootstrap calls.
        """
        rng = random.Random(202611)
        groups = [
            ("TUBE_G01", "BATCH_2026_01", "VID_G01", "yellow"),
            ("TUBE_G02", "BATCH_2026_02", "VID_G02", "blue"),
            ("TUBE_G03", "BATCH_2026_03", "VID_G03", "yellow"),
            ("TUBE_G04", "BATCH_2026_04", "VID_G04", "blue"),
        ]
        group_idx_by_tube = {g[0]: idx for idx, g in enumerate(groups)}

        distinct_sources: list[tuple[np.ndarray, str]] = []
        seen_digests: set[str] = set()
        raw_sources: list[np.ndarray] = []
        if live_crops:
            raw_sources.extend(live_crops)
        elif live_crop_bgr is not None:
            raw_sources.append(live_crop_bgr)

        for src in raw_sources:
            if src is None or not isinstance(src, np.ndarray) or src.size == 0:
                continue
            digest = hashlib.sha256(np.ascontiguousarray(src).tobytes()).hexdigest()
            if digest not in seen_digests:
                seen_digests.add(digest)
                distinct_sources.append((src, digest))

        # Inspect existing manifest lineage so repeated bootstrap calls keep each source pinned to one group
        existing_source_to_groups: dict[str, set[str]] = defaultdict(set)
        existing_group_to_sources: dict[str, set[str]] = defaultdict(set)
        with self._lock:
            for existing_row in self._read_manifest_rows():
                src_dig = (existing_row.get("source_digest") or "").strip()
                tk = (existing_row.get("physical_tube_id") or "").strip()
                if src_dig and tk and not _is_ungrouped_tube(tk):
                    existing_source_to_groups[src_dig].add(tk)
                    existing_group_to_sources[tk].add(src_dig)

        group_sources: list[tuple[np.ndarray, str] | None] = [None] * len(groups)
        unpinned_sources: list[tuple[np.ndarray, str]] = []
        for src, digest in distinct_sources:
            prior_groups = existing_source_to_groups.get(digest)
            if prior_groups:
                # Pin to the existing group if uniquely within our bootstrap groups; never assign to a second group
                if len(prior_groups) == 1:
                    pinned_tube = next(iter(prior_groups))
                    if pinned_tube in group_idx_by_tube:
                        slot = group_idx_by_tube[pinned_tube]
                        if group_sources[slot] is None:
                            group_sources[slot] = (src, digest)
                continue
            unpinned_sources.append((src, digest))

        # Assign new distinct sources to unoccupied slots, preferring groups with no prior source digest
        open_slots = [idx for idx in range(len(groups)) if group_sources[idx] is None]
        open_slots.sort(key=lambda idx: (len(existing_group_to_sources.get(groups[idx][0], set())), idx))
        for (src, digest), slot in zip(unpinned_sources, open_slots, strict=False):
            group_sources[slot] = (src, digest)

        live_crop_groups = [groups[idx][0] for idx, item in enumerate(group_sources) if item is not None]

        ds_dir = self._dataset_dir()
        created = 0
        pending_rows: list[dict[str, str]] = []
        pending_coco: list[dict[str, Any]] = []

        for g_idx, (tube_id, batch_id, vid_id, p_color) in enumerate(groups):
            slot_item = group_sources[g_idx]
            group_base_bgr = slot_item[0] if slot_item is not None else None
            group_source_digest = slot_item[1] if slot_item is not None else ""
            for label in CLASSES:
                for k in range(3):
                    variant = g_idx * 10 + k
                    img, box_norm = self._synthesize_roi_sample_with_box(group_base_bgr, label, p_color, variant, rng)
                    img_dir = ds_dir / "images" / label
                    img_dir.mkdir(parents=True, exist_ok=True)
                    sample_id = f"STD_{tube_id}_{label[:4].upper()}_{k + 1:02d}_{uuid.uuid4().hex[:4].upper()}"
                    rel_path = Path("images") / label / f"{sample_id}.jpg"
                    abs_path = ds_dir / rel_path
                    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
                    if ok:
                        abs_path.write_bytes(bytes(buf))
                        img_h, img_w = int(img.shape[0]), int(img.shape[1])
                        box_csv = ",".join(f"{v:.4f}" for v in box_norm)
                        pending_rows.append(
                            {
                                "sample_id": sample_id,
                                "path": rel_path.as_posix(),
                                "label": label,
                                "physical_tube_id": tube_id,
                                "batch_id": batch_id,
                                "video_group_id": vid_id,
                                "pipe_color": p_color,
                                "box_xyxy": box_csv,
                                "crop_box_xyxy": box_csv,
                                "source_digest": group_source_digest,
                                "captured_at_utc": _utc_now(),
                            }
                        )
                        pending_coco.append(
                            {
                                "sample_id": sample_id,
                                "rel_path": rel_path.as_posix(),
                                "label": label,
                                "crop_box_xyxy": box_norm,
                                "image_width": img_w,
                                "image_height": img_h,
                            }
                        )
                        created += 1

        # Add isolated synthetic challenge samples without blending any live source crop
        for k in range(4):
            img, box_norm = self._synthesize_roi_sample_with_box(None, CHALLENGE, "yellow", 90 + k, rng)
            img_dir = ds_dir / "images" / CHALLENGE
            img_dir.mkdir(parents=True, exist_ok=True)
            sample_id = f"STD_CHAL_{k + 1:02d}_{uuid.uuid4().hex[:4].upper()}"
            rel_path = Path("images") / CHALLENGE / f"{sample_id}.jpg"
            abs_path = ds_dir / rel_path
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
            if ok:
                abs_path.write_bytes(bytes(buf))
                img_h, img_w = int(img.shape[0]), int(img.shape[1])
                box_csv = ",".join(f"{v:.4f}" for v in box_norm)
                pending_rows.append(
                    {
                        "sample_id": sample_id,
                        "path": rel_path.as_posix(),
                        "label": CHALLENGE,
                        "physical_tube_id": f"TUBE_CHAL_ISO_{k + 1:02d}",
                        "batch_id": f"BATCH_CHAL_ISO_{k + 1:02d}",
                        "video_group_id": f"VID_CHAL_ISO_{k + 1:02d}",
                        "pipe_color": "yellow",
                        "box_xyxy": box_csv,
                        "crop_box_xyxy": box_csv,
                        "source_digest": "",
                        "captured_at_utc": _utc_now(),
                    }
                )
                pending_coco.append(
                    {
                        "sample_id": sample_id,
                        "rel_path": rel_path.as_posix(),
                        "label": CHALLENGE,
                        "crop_box_xyxy": box_norm,
                        "image_width": img_w,
                        "image_height": img_h,
                    }
                )
                created += 1

        with self._lock:
            for row in pending_rows:
                self._append_manifest_row(row)
            coco_synced = self._append_coco_annotations(pending_coco)

        summary = self.get_dataset_summary()
        summary["bootstrapped_count"] = created
        summary["coco_synced"] = coco_synced
        summary["is_synthetic_bootstrap"] = True
        summary["dataset_origin"] = "synthetic_smoke_test"
        summary["live_crop_groups"] = live_crop_groups
        return summary

    @classmethod
    def _row_image_digest(cls, row: dict[str, str]) -> str:
        cached = (row.get("image_sha256") or "").strip()
        if cached:
            return cached
        raw_abs = (row.get("abs_path") or "").strip()
        if not raw_abs:
            return ""
        p = Path(raw_abs)
        if not p.is_file():
            return ""
        try:
            digest = _sha256_file(p)
        except OSError:
            return ""
        row["image_sha256"] = digest
        return digest

    @classmethod
    def _select_group_split(
        cls,
        rows: list[dict[str, str]],
        forbidden_holdout_groups: set[str] | None = None,
        forbidden_holdout_sample_ids: set[str] | None = None,
        forbidden_holdout_image_digests: set[str] | None = None,
        forbidden_holdout_source_digests: set[str] | None = None,
    ) -> dict[str, Any]:
        """Select a leak-free group split with O(1) per-pair coverage checks and group/sample/image/source provenance isolation."""
        by_tube: dict[str, list[dict[str, str]]] = defaultdict(list)
        ungrouped_rows: list[dict[str, str]] = []
        challenges: list[dict[str, str]] = []
        for r in rows:
            lbl = r.get("label") or ""
            if lbl == CHALLENGE:
                challenges.append(r)
            elif lbl in CLASSES:
                tk = r.get("physical_tube_id") or ""
                if _is_ungrouped_tube(tk):
                    ungrouped_rows.append(r)
                else:
                    by_tube[tk].append(r)

        tube_keys = sorted(by_tube.keys())
        if len(tube_keys) < 3:
            raise ValueError("独立料管分组不足 3 组，无法构建无泄漏的 train/val/test 划分（未分组样本不计入独立料管组）")

        forbidden_groups = {g for g in (forbidden_holdout_groups or set()) if not _is_ungrouped_tube(g)}
        forbidden_samples = {s.strip() for s in (forbidden_holdout_sample_ids or set()) if isinstance(s, str) and s.strip()}
        forbidden_images = {d.strip().lower() for d in (forbidden_holdout_image_digests or set()) if isinstance(d, str) and d.strip()}
        forbidden_sources = {s.strip().lower() for s in (forbidden_holdout_source_digests or set()) if isinstance(s, str) and s.strip()}

        # Track source_digest occurrences across all groups + ungrouped so one source can never span multiple partitions (R2-15)
        tube_sources: dict[str, set[str]] = {}
        source_occurrences: dict[str, int] = defaultdict(int)
        for tk in tube_keys:
            srcs = {(r.get("source_digest") or "").strip().lower() for r in by_tube[tk] if (r.get("source_digest") or "").strip()}
            tube_sources[tk] = srcs
            for src in srcs:
                source_occurrences[src] += 1
        ungrouped_sources = {
            (r.get("source_digest") or "").strip().lower()
            for r in ungrouped_rows
            if (r.get("source_digest") or "").strip()
        }
        for src in ungrouped_sources:
            source_occurrences[src] += 1

        ineligible_tubes: set[str] = set()
        for tk in tube_keys:
            if tk in forbidden_groups:
                ineligible_tubes.add(tk)
                continue
            if any(source_occurrences[src] > 1 for src in tube_sources[tk]):
                ineligible_tubes.add(tk)
                continue
            if forbidden_sources and (tube_sources[tk] & forbidden_sources):
                ineligible_tubes.add(tk)
                continue
            if forbidden_samples and any((r.get("sample_id") or "").strip() in forbidden_samples for r in by_tube[tk]):
                ineligible_tubes.add(tk)
                continue
            if forbidden_images and any(cls._row_image_digest(r).lower() in forbidden_images for r in by_tube[tk]):
                ineligible_tubes.add(tk)
                continue

        holdout_candidates = [tk for tk in tube_keys if tk not in ineligible_tubes]
        if len(holdout_candidates) < 2:
            raise ValueError(
                "可用于验证/测试留出的未训练独立料管组不足 2 组"
                + (f"（历史已训练/共享源分组：{', '.join(sorted(ineligible_tubes))}）" if ineligible_tubes else "")
            )

        tube_labels = {tk: {r["label"] for r in by_tube[tk]} for tk in tube_keys}
        ungrouped_labels = {r["label"] for r in ungrouped_rows if r.get("label") in CLASSES}
        # Precompute per-class independent-group counts once (R2-34) and count train-only ungrouped labels (R2-16)
        class_group_counts = {c: sum(1 for tk in tube_keys if c in tube_labels[tk]) for c in CLASSES}
        classes_needed_from_tubes = [c for c in CLASSES if c not in ungrouped_labels]

        def _pair_is_valid(val_tk: str, test_tk: str) -> bool:
            if len(tube_keys) <= 2:
                return False
            v_lbls = tube_labels[val_tk]
            t_lbls = tube_labels[test_tk]
            for c in classes_needed_from_tubes:
                remaining = class_group_counts[c] - (1 if c in v_lbls else 0) - (1 if c in t_lbls else 0)
                if remaining <= 0:
                    return False
            return True

        chosen: tuple[str, str] | None = None
        full_holdouts = [tk for tk in holdout_candidates if len(tube_labels[tk]) == len(CLASSES)]
        if len(full_holdouts) >= 2 and _pair_is_valid(full_holdouts[-2], full_holdouts[-1]):
            chosen = (full_holdouts[-2], full_holdouts[-1])

        if chosen is None:
            best_score: tuple[int, int, int, int] | None = None
            for i, val_tk in enumerate(holdout_candidates):
                v_cnt = len(tube_labels[val_tk])
                for j in range(i + 1, len(holdout_candidates)):
                    test_tk = holdout_candidates[j]
                    if not _pair_is_valid(val_tk, test_tk):
                        continue
                    t_cnt = len(tube_labels[test_tk])
                    score = (v_cnt + t_cnt, min(v_cnt, t_cnt), i, j)
                    if best_score is None or score > best_score:
                        best_score = score
                        chosen = (val_tk, test_tk)

        if chosen is None:
            deficient = [c for c in classes_needed_from_tubes if class_group_counts[c] < 3]
            if not deficient:
                fallback_val, fallback_test = holdout_candidates[-2], holdout_candidates[-1]
                v_lbls = tube_labels[fallback_val]
                t_lbls = tube_labels[fallback_test]
                deficient = [
                    c
                    for c in classes_needed_from_tubes
                    if class_group_counts[c] - (1 if c in v_lbls else 0) - (1 if c in t_lbls else 0) <= 0
                ] or list(CLASSES)
            raise ValueError(
                f"当前独立料管分组布局无法构建覆盖全部 4 类的训练集，以下类别需要更多独立料管组：{', '.join(deficient)}"
            )

        val_tube, test_tube = chosen
        train_groups: list[str] = []
        train_rows: list[dict[str, str]] = []
        val_rows: list[dict[str, str]] = []
        test_rows: list[dict[str, str]] = []
        for tk in tube_keys:
            if tk == val_tube:
                val_rows.extend(by_tube[tk])
            elif tk == test_tube:
                test_rows.extend(by_tube[tk])
            else:
                train_groups.append(tk)
                train_rows.extend(by_tube[tk])

        # Keep ungrouped samples together strictly inside train_rows (never in independent holdouts)
        train_rows.extend(ungrouped_rows)

        actual_train_classes = {r["label"] for r in train_rows}
        missing_in_train = [c for c in CLASSES if c not in actual_train_classes]
        if missing_in_train:
            raise ValueError(
                f"分组划分校验未通过，训练集缺少以下类别：{', '.join(missing_in_train)}"
            )
        if not val_rows or not test_rows:
            raise ValueError("分组划分校验未通过，验证集或测试集为空")
        if val_tube in ineligible_tubes or test_tube in ineligible_tubes:
            raise ValueError("分组划分校验未通过，验证或测试集包含不可留出分组")

        return {
            "train_rows": train_rows,
            "val_rows": val_rows,
            "test_rows": test_rows,
            "challenges": challenges,
            "train_groups": sorted(train_groups),
            "val_groups": [val_tube],
            "test_groups": [test_tube],
            "ungrouped_count": len(ungrouped_rows),
        }

    def _build_dataset_summary(self, rows: list[dict[str, str]]) -> dict[str, Any]:
        counts = {label: 0 for label in ALL_LABELS}
        tubes: set[str] = set()
        batches: set[str] = set()
        ungrouped_count = 0
        for r in rows:
            lbl = r.get("label", "")
            if lbl in counts:
                counts[lbl] += 1
            if lbl in CLASSES:
                tk = r.get("physical_tube_id") or ""
                if _is_ungrouped_tube(tk):
                    ungrouped_count += 1
                else:
                    tubes.add(tk)
                bid = (r.get("batch_id") or "").strip()
                if bid and bid.upper() != UNGROUPED_BATCH_ID:
                    batches.add(bid)

        all_classes_present = all(counts[c] >= 3 for c in CLASSES)
        enough_groups = len(tubes) >= 3
        split_ok = False
        split_error: str | None = None
        if all_classes_present and enough_groups:
            try:
                self._select_group_split(rows)
                split_ok = True
            except ValueError as exc:
                split_error = str(exc)

        ready = bool(all_classes_present and enough_groups and split_ok)
        if ready:
            reason = "数据集已满足分组隔离与四类均衡要求，可随时启动训练"
        elif split_error:
            reason = split_error
        else:
            reason = "需至少 3 个独立料管组且每类（完整/划除/遮挡/脱落）至少 3 张样本；可点击「一键生成标准训练集」或实时抓拍补充"

        recent = []
        for r in reversed(rows[-16:]):
            recent.append(
                {
                    "sample_id": r.get("sample_id"),
                    "label": r.get("label"),
                    "physical_tube_id": r.get("physical_tube_id"),
                    "batch_id": r.get("batch_id"),
                    "pipe_color": r.get("pipe_color") or "yellow",
                    "captured_at_utc": r.get("captured_at_utc"),
                    "image_url": f"/api/dataset/sample/{r.get('sample_id')}/image",
                }
            )

        return {
            "total_samples": len(rows),
            "class_counts": counts,
            "independent_tube_groups": len(tubes),
            "independent_batches": len(batches),
            "ungrouped_samples": ungrouped_count,
            "ready_to_train": ready,
            "readiness_reason": reason,
            "manifest_path": str(self._manifest_path()),
            "recent_samples": recent,
        }

    def get_dataset_summary(self) -> dict[str, Any]:
        with self._lock:
            rows = self._read_manifest_rows()
        return self._build_dataset_summary(rows)

    def find_verified_onnx_runs(self) -> list[dict[str, Any]]:
        """Return newest-first ONNX runs that passed parity verification and match their recorded SHA-256 digest."""
        runs_dir = self._runs_dir()
        if not runs_dir.is_dir():
            return []
        candidates = sorted(
            runs_dir.glob("*/label_classifier.onnx"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        verified: list[dict[str, Any]] = []
        for onnx_path in candidates:
            metrics_path = onnx_path.parent / "metrics.json"
            if not metrics_path.is_file():
                continue
            try:
                meta = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Skipping ONNX run %s due to unreadable metrics.json: %s", onnx_path.parent.name, exc)
                continue
            if not isinstance(meta, dict):
                continue
            parity_ok = meta["onnx_parity_ok"] if "onnx_parity_ok" in meta else meta.get("onnx_parity_passed")
            if parity_ok is not True:
                continue
            try:
                actual_sha = _sha256_file(onnx_path)
            except OSError as exc:
                logger.warning("Skipping unreadable ONNX artifact %s: %s", onnx_path, exc)
                continue
            recorded_sha = meta.get("onnx_sha256")
            if isinstance(recorded_sha, str) and recorded_sha.strip() and recorded_sha.strip().lower() != actual_sha.lower():
                logger.warning("Skipping ONNX artifact %s due to SHA-256 digest mismatch", onnx_path)
                continue
            verified.append(
                {
                    "model_version": onnx_path.parent.name,
                    "onnx_path": str(onnx_path),
                    "onnx_sha256": actual_sha,
                    "input_size": int(meta.get("input_size") or 96),
                    "metrics": meta,
                }
            )
        return verified

    def _allocate_run_dir(self) -> tuple[str, Path]:
        runs_dir = self._runs_dir()
        base_id = "mnetv3_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = base_id
        run_dir = runs_dir / run_id
        while run_dir.exists():
            run_id = f"{base_id}_{uuid.uuid4().hex[:4]}"
            run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_id, run_dir

    def _snapshot_manifest_for_run(self, rows: list[dict[str, str]], run_dir: Path) -> tuple[list[dict[str, str]], Path]:
        """Copy selected dataset images into a run-scoped immutable snapshot directory and hash each image (called under self._lock)."""
        snapshot_dir = run_dir / "_input_snapshot"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshotted: list[dict[str, str]] = []
        for idx, r in enumerate(rows):
            src = Path(r["abs_path"])
            if not src.is_file():
                continue
            suffix = src.suffix or ".jpg"
            safe_sid = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(r.get("sample_id") or f"row{idx}"))
            dst = snapshot_dir / f"{idx:05d}_{safe_sid}{suffix}"
            img_bytes = src.read_bytes()
            img_sha = hashlib.sha256(img_bytes).hexdigest()
            dst.write_bytes(img_bytes)
            snapshotted.append({**r, "abs_path": str(dst), "image_sha256": img_sha})
        return snapshotted, snapshot_dir

    @staticmethod
    def _extract_str_set(container: dict[str, Any], keys: tuple[str, ...], *, exclude_ungrouped: bool = False) -> set[str]:
        for key in keys:
            raw = container.get(key)
            if isinstance(raw, list) and raw and all(isinstance(x, str) and x.strip() for x in raw):
                vals = {x.strip() for x in raw if not (exclude_ungrouped and _is_ungrouped_tube(x))}
                if vals:
                    return vals
        return set()

    @classmethod
    def _read_checkpoint_provenance(cls, pth_candidate: Path, ckpt: dict[str, Any]) -> dict[str, set[str]] | None:
        """Extract previously trained group, sample ID, image digest, and source lineage from checkpoint or sibling metrics.json."""
        meta: dict[str, Any] = {}
        metrics_path = pth_candidate.parent / "metrics.json"
        if metrics_path.is_file():
            try:
                loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    meta = loaded
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                logger.warning("Failed to read provenance metrics for checkpoint %s: %s", pth_candidate, exc)

        seen_groups = cls._extract_str_set(ckpt, ("seen_groups", "train_groups"), exclude_ungrouped=True) or cls._extract_str_set(
            meta, ("seen_groups", "train_groups"), exclude_ungrouped=True
        )
        seen_samples = cls._extract_str_set(ckpt, ("seen_sample_ids", "train_sample_ids")) or cls._extract_str_set(
            meta, ("seen_sample_ids", "train_sample_ids")
        )
        seen_images = {
            d.lower()
            for d in (
                cls._extract_str_set(ckpt, ("seen_image_digests", "train_image_digests"))
                or cls._extract_str_set(meta, ("seen_image_digests", "train_image_digests"))
            )
        }
        seen_sources = {
            s.lower()
            for s in (
                cls._extract_str_set(ckpt, ("seen_source_digests", "train_source_digests"))
                or cls._extract_str_set(meta, ("seen_source_digests", "train_source_digests"))
            )
        }

        if not seen_groups and not seen_samples and not seen_images:
            return None
        return {
            "seen_groups": seen_groups,
            "seen_sample_ids": seen_samples,
            "seen_image_digests": seen_images,
            "seen_source_digests": seen_sources,
        }

    @classmethod
    def _read_checkpoint_seen_groups(cls, pth_candidate: Path, ckpt: dict[str, Any]) -> set[str] | None:
        prov = cls._read_checkpoint_provenance(pth_candidate, ckpt)
        if prov is None or not prov["seen_groups"]:
            return None
        return prov["seen_groups"]

    def _persist_metrics_with_retry(self, metrics_path: Path, payload: dict[str, Any], attempts: int = 3) -> str | None:
        """Persist metrics.json atomically with bounded retry; never raises so confirmed model activation is preserved (R2-18)."""
        last_err: str | None = None
        disk_payload = {**payload, "metrics_persisted": True, "metrics_write_error": None}
        for attempt in range(max(1, attempts)):
            try:
                _write_json_atomic(metrics_path, disk_payload)
                with self._lock:
                    self._pending_metrics_write = None
                return None
            except OSError as exc:
                last_err = str(exc) or exc.__class__.__name__
                logger.warning(
                    "Failed to write metrics.json (%s, attempt %d/%d): %s",
                    metrics_path,
                    attempt + 1,
                    attempts,
                    last_err,
                )
                if attempt + 1 < attempts:
                    time.sleep(0.05)
        with self._lock:
            self._pending_metrics_write = (metrics_path, disk_payload)
        return last_err

    def retry_persist_latest_metrics(self) -> bool:
        """Retry writing the most recent failed metrics.json update without altering active model status."""
        with self._lock:
            pending = self._pending_metrics_write
        if pending is None:
            return True
        metrics_path, payload = pending
        err = self._persist_metrics_with_retry(metrics_path, payload, attempts=1)
        if err is None:
            with self._lock:
                self._status["metrics_persisted"] = True
                self._status["metrics_write_error"] = None
                if isinstance(self._status.get("latest_run"), dict):
                    self._status["latest_run"]["metrics_persisted"] = True
                    self._status["latest_run"]["metrics_write_error"] = None
                self._status["updated_at_utc"] = _utc_now()
            return True
        return False

    def get_training_status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def start_training(
        self,
        epochs: int = 8,
        input_size: int = 96,
        batch_size: int = 8,
        learning_rate: float = 0.0025,
        resume_from_latest: bool = True,
        use_augmentation: bool = True,
        on_complete: Callable[[dict[str, Any]], Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._status["state"] in {"TRAINING", "EXPORTING_ONNX"}:
                raise RuntimeError("训练任务正在运行中，请等待当前训练完成")

            rows = self._read_manifest_rows()
            summary = self._build_dataset_summary(rows)
            if not summary["ready_to_train"]:
                raise ValueError(summary["readiness_reason"])

            run_id, run_dir = self._allocate_run_dir()
            try:
                snapshotted_rows, snapshot_dir = self._snapshot_manifest_for_run(rows, run_dir)
                # Re-verify split on snapshotted immutable files before committing to TRAINING state
                self._select_group_split(snapshotted_rows)
            except Exception:
                shutil.rmtree(run_dir, ignore_errors=True)
                raise

            mode_desc = "增量微调 (Fine-tune)" if resume_from_latest else "从零初始化训练"
            self._pending_metrics_write = None
            self._status = {
                "state": "TRAINING",
                "message": f"正在执行分组防泄漏审计与启动 {mode_desc}…",
                "current_epoch": 0,
                "total_epochs": epochs,
                "progress_pct": 5,
                "history": [],
                "latest_run": None,
                "exported": False,
                "activated": False,
                "activation_status": None,
                "metrics_persisted": True,
                "metrics_write_error": None,
                "error": None,
                "updated_at_utc": _utc_now(),
            }

        self._thread = threading.Thread(
            target=self._run_training_worker,
            args=(
                epochs,
                input_size,
                batch_size,
                learning_rate,
                resume_from_latest,
                use_augmentation,
                on_complete,
                run_id,
                run_dir,
                snapshotted_rows,
                snapshot_dir,
            ),
            name="MobileNetV3Trainer",
            daemon=True,
        )
        self._thread.start()
        return self.get_training_status()

    def _run_training_worker(
        self,
        epochs: int,
        input_size: int,
        batch_size: int,
        learning_rate: float,
        resume_from_latest: bool,
        use_augmentation: bool,
        on_complete: Callable[[dict[str, Any]], Any] | None,
        run_id: str | None = None,
        run_dir: Path | None = None,
        snapshotted_rows: list[dict[str, str]] | None = None,
        snapshot_dir: Path | None = None,
    ) -> None:
        staged_onnx_path: Path | None = None
        try:
            import onnxruntime as ort
            import torch
            from PIL import Image
            from torch import nn
            from torch.utils.data import DataLoader, Dataset
            from torchvision import transforms
            from torchvision.models import mobilenet_v3_small

            manifest_csv = self._manifest_path()
            if snapshotted_rows is None or run_dir is None or run_id is None:
                with self._lock:
                    live_rows = self._read_manifest_rows()
                    run_id, run_dir = self._allocate_run_dir()
                    snapshotted_rows, snapshot_dir = self._snapshot_manifest_for_run(live_rows, run_dir)

            rows = snapshotted_rows

            random.seed(2026)
            np.random.seed(2026)
            torch.manual_seed(2026)
            device = torch.device("cpu")

            base_mnet = mobilenet_v3_small(weights=None)
            base_mnet.classifier[-1] = nn.Linear(base_mnet.classifier[-1].in_features, len(CLASSES))
            for p in base_mnet.features.parameters():
                p.requires_grad_(False)

            class _FastConvergingMobileNetV3(nn.Module):
                """MobileNetV3-Small with a BatchNorm-free multi-scale spatial-contrast head for deterministic small-sample convergence."""

                def __init__(self, backbone: nn.Module, num_classes: int) -> None:
                    super().__init__()
                    self.backbone = backbone
                    self.spatial_branch = nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.AdaptiveAvgPool2d((4, 4)),
                        nn.Flatten(),
                        nn.LayerNorm(48 * 16),
                        nn.Linear(48 * 16, 96),
                        nn.ReLU(inplace=True),
                        nn.Linear(96, num_classes),
                    )

                def forward(self, x: Any) -> Any:
                    return self.backbone(x) * 0.01 + self.spatial_branch(x)

            model = _FastConvergingMobileNetV3(base_mnet, len(CLASSES)).to(device)
            resumed_from: str | None = None
            resumed_seen_groups: set[str] = set()
            resumed_seen_sample_ids: set[str] = set()
            resumed_seen_image_digests: set[str] = set()
            resumed_seen_source_digests: set[str] = set()
            split: dict[str, Any] | None = None

            if resume_from_latest:
                prev_pths = sorted(
                    (p for p in self._runs_dir().glob("*/label_classifier.pth") if p.parent != run_dir),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                provenance_rejections: list[str] = []
                for pth_candidate in prev_pths:
                    try:
                        ckpt = torch.load(pth_candidate, map_location="cpu", weights_only=True)
                        if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
                            continue
                        if int(ckpt.get("input_size") or input_size) != input_size:
                            continue
                    except Exception as exc:
                        logger.debug("Skipping incompatible checkpoint %s: %s", pth_candidate, exc)
                        continue

                    prior_prov = self._read_checkpoint_provenance(pth_candidate, ckpt)
                    if prior_prov is None:
                        logger.warning(
                            "Rejecting warm-start checkpoint %s without training group provenance",
                            pth_candidate.parent.name,
                        )
                        provenance_rejections.append(f"{pth_candidate.parent.name}(缺少历史训练分组溯源)")
                        continue

                    try:
                        candidate_split = self._select_group_split(
                            rows,
                            forbidden_holdout_groups=prior_prov["seen_groups"],
                            forbidden_holdout_sample_ids=prior_prov["seen_sample_ids"],
                            forbidden_holdout_image_digests=prior_prov["seen_image_digests"],
                            forbidden_holdout_source_digests=prior_prov["seen_source_digests"],
                        )
                    except ValueError as split_exc:
                        logger.warning(
                            "Rejecting warm-start checkpoint %s due to holdout leakage risk: %s",
                            pth_candidate.parent.name,
                            split_exc,
                        )
                        provenance_rejections.append(f"{pth_candidate.parent.name}({split_exc})")
                        continue

                    try:
                        model.load_state_dict(ckpt["state_dict"], strict=True)
                    except Exception as exc:
                        logger.debug("Skipping incompatible state_dict %s: %s", pth_candidate, exc)
                        continue

                    resumed_from = pth_candidate.parent.name
                    resumed_seen_groups = prior_prov["seen_groups"]
                    resumed_seen_sample_ids = prior_prov["seen_sample_ids"]
                    resumed_seen_image_digests = prior_prov["seen_image_digests"]
                    resumed_seen_source_digests = prior_prov["seen_source_digests"]
                    split = candidate_split
                    break

                if resumed_from is None and provenance_rejections:
                    raise ValueError(
                        "无法在不泄漏验证/测试留出集的前提下继续增量微调："
                        + "；".join(provenance_rejections)
                        + "。请补充未训练过的独立料管组，或关闭增量微调从零训练。"
                    )

            if split is None:
                split = self._select_group_split(rows)

            train_rows: list[dict[str, str]] = split["train_rows"]
            val_rows: list[dict[str, str]] = split["val_rows"]
            test_rows: list[dict[str, str]] = split["test_rows"]
            challenges: list[dict[str, str]] = split["challenges"]
            train_groups: list[str] = split["train_groups"]
            val_groups: list[str] = split["val_groups"]
            test_groups: list[str] = split["test_groups"]
            train_sample_ids = sorted({(r.get("sample_id") or "").strip() for r in train_rows if (r.get("sample_id") or "").strip()})
            train_image_digests = sorted({self._row_image_digest(r).lower() for r in train_rows if self._row_image_digest(r)})
            train_source_digests = sorted(
                {(r.get("source_digest") or "").strip().lower() for r in train_rows if (r.get("source_digest") or "").strip()}
            )
            cumulative_seen_groups = sorted(resumed_seen_groups | set(train_groups))
            cumulative_seen_sample_ids = sorted(resumed_seen_sample_ids | set(train_sample_ids))
            cumulative_seen_image_digests = sorted(resumed_seen_image_digests | set(train_image_digests))
            cumulative_seen_source_digests = sorted(resumed_seen_source_digests | set(train_source_digests))

            holdout_rows = [*val_rows, *test_rows]
            if resumed_seen_groups and not (
                set(val_groups).isdisjoint(resumed_seen_groups) and set(test_groups).isdisjoint(resumed_seen_groups)
            ):
                raise RuntimeError("Warm-start holdout leakage detected: previously trained group placed in holdout")
            if resumed_seen_sample_ids and any((r.get("sample_id") or "").strip() in resumed_seen_sample_ids for r in holdout_rows):
                raise RuntimeError("Warm-start holdout leakage detected: previously trained sample placed in holdout")
            if resumed_seen_image_digests and any(self._row_image_digest(r).lower() in resumed_seen_image_digests for r in holdout_rows):
                raise RuntimeError("Warm-start holdout leakage detected: previously trained image placed in holdout")
            if resumed_seen_source_digests and any(
                (r.get("source_digest") or "").strip().lower() in resumed_seen_source_digests for r in holdout_rows
            ):
                raise RuntimeError("Warm-start holdout leakage detected: previously trained source placed in holdout")

            eval_tf = transforms.Compose(
                [
                    transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
                ]
            )
            if use_augmentation:
                train_tf = transforms.Compose(
                    [
                        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BILINEAR),
                        transforms.ColorJitter(brightness=0.22, contrast=0.22, saturation=0.18),
                        transforms.RandomAffine(degrees=5, translate=(0.04, 0.04)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
                    ]
                )
            else:
                train_tf = eval_tf

            class _RoiSet(Dataset):
                def __init__(self, items: list[dict[str, str]]) -> None:
                    self.items = items

                def __len__(self) -> int:
                    return len(self.items)

                def __getitem__(self, idx: int) -> tuple[Any, int]:
                    row = self.items[idx]
                    with Image.open(row["abs_path"]) as im:
                        x = train_tf(im.convert("RGB"))
                    return x, CLASSES.index(row["label"])

            def _eval_split(split_rows: list[dict[str, str]]) -> dict[str, Any]:
                model.eval()
                matrix = [[0 for _ in CLASSES] for _ in CLASSES]
                with torch.inference_mode():
                    for i in range(0, len(split_rows), batch_size):
                        batch = split_rows[i : i + batch_size]
                        tensors = []
                        for r in batch:
                            with Image.open(r["abs_path"]) as im:
                                tensors.append(eval_tf(im.convert("RGB")))
                        x = torch.stack(tensors).to(device)
                        probs = torch.softmax(model(x), dim=1).cpu()
                        preds = probs.argmax(dim=1).tolist()
                        for r, p in zip(batch, preds, strict=True):
                            t_idx = CLASSES.index(r["label"])
                            matrix[t_idx][int(p)] += 1
                correct = sum(matrix[i][i] for i in range(len(CLASSES)))
                acc = correct / max(1, len(split_rows))
                complete_idx = CLASSES.index("complete")
                false_accept = sum(matrix[complete_idx][j] for j in range(len(CLASSES)) if j != complete_idx)
                return {
                    "samples": len(split_rows),
                    "accuracy": round(acc, 4),
                    "confusion_matrix": {"labels": CLASSES, "matrix": matrix},
                    "complete_false_accept_total": false_accept,
                }

            loader = DataLoader(_RoiSet(train_rows), batch_size=batch_size, shuffle=True, num_workers=0)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            loss_fn = nn.CrossEntropyLoss()

            history: list[dict[str, Any]] = []
            best_acc = -1.0
            best_state = None
            t_start = time.time()

            for ep in range(1, epochs + 1):
                model.train()
                total_loss = 0.0
                seen = 0
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    loss = loss_fn(logits, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.item()) * len(labels)
                    seen += len(labels)

                val_res = _eval_split(val_rows)
                ep_item = {
                    "epoch": ep,
                    "train_loss": round(total_loss / max(1, seen), 4),
                    "val_accuracy": val_res["accuracy"],
                }
                history.append(ep_item)
                if val_res["accuracy"] >= best_acc:
                    best_acc = val_res["accuracy"]
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                with self._lock:
                    self._status.update(
                        {
                            "state": "TRAINING",
                            "message": f"正在训练 MobileNetV3-Small：第 {ep}/{epochs} 轮 (Loss: {ep_item['train_loss']:.4f}, 验证集准确率: {val_res['accuracy'] * 100:.1f}%)",
                            "current_epoch": ep,
                            "total_epochs": epochs,
                            "progress_pct": int(10 + (ep / epochs) * 70),
                            "history": list(history),
                            "updated_at_utc": _utc_now(),
                        }
                    )

            if best_state is not None:
                model.load_state_dict(best_state)

            with self._lock:
                self._status.update(
                    {
                        "state": "EXPORTING_ONNX",
                        "message": "正在导出 ONNX 推理模型并校验 PyTorch / ONNX Runtime 数值一致性…",
                        "progress_pct": 88,
                        "updated_at_utc": _utc_now(),
                    }
                )

            pth_path = run_dir / "label_classifier.pth"
            staged_onnx_path = run_dir / "label_classifier.unverified.onnx"
            onnx_path = run_dir / "label_classifier.onnx"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "input_size": input_size,
                    "train_groups": train_groups,
                    "seen_groups": cumulative_seen_groups,
                    "train_sample_ids": train_sample_ids,
                    "seen_sample_ids": cumulative_seen_sample_ids,
                    "train_image_digests": train_image_digests,
                    "seen_image_digests": cumulative_seen_image_digests,
                    "train_source_digests": train_source_digests,
                    "seen_source_digests": cumulative_seen_source_digests,
                    "val_groups": val_groups,
                    "test_groups": test_groups,
                    "resumed_from": resumed_from,
                },
                pth_path,
            )

            model.eval().cpu()
            sample_tensors = []
            for r in test_rows[: min(8, len(test_rows))]:
                with Image.open(r["abs_path"]) as im:
                    sample_tensors.append(eval_tf(im.convert("RGB")))
            example = torch.stack(sample_tensors)

            torch.onnx.export(
                model,
                example,
                str(staged_onnx_path),
                input_names=["image"],
                output_names=["logits"],
                dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )

            ort_sess = ort.InferenceSession(str(staged_onnx_path), providers=["CPUExecutionProvider"])
            ort_out = ort_sess.run(["logits"], {"image": example.numpy()})[0]
            with torch.inference_mode():
                pt_out = model(example).numpy()
            max_abs_diff = float(np.max(np.abs(pt_out - ort_out)))
            parity_ok = bool(
                np.all(np.isfinite(pt_out))
                and np.all(np.isfinite(ort_out))
                and max_abs_diff < ONNX_PARITY_MAX_ABS_DIFF
            )

            test_res = _eval_split(test_rows)
            val_final = _eval_split(val_rows)
            pth_sha = _sha256_file(pth_path)
            elapsed_s = round(time.time() - t_start, 2)

            holdout_has_synthetic = any(
                str(r.get("sample_id") or "").startswith("STD_") for r in (*val_rows, *test_rows)
            )
            any_synthetic = holdout_has_synthetic or any(
                str(r.get("sample_id") or "").startswith("STD_") for r in train_rows
            )
            evaluation_kind = "synthetic_smoke_test" if holdout_has_synthetic else "independent_holdout"
            independent_holdout_verified = bool(
                not holdout_has_synthetic
                and all(not _is_ungrouped_tube(g) for g in (*val_groups, *test_groups))
            )

            run_summary: dict[str, Any] = {
                "model_name": "mobilenetv3-small-igbt-roi",
                "model_version": run_id,
                "architecture": "torchvision.models.mobilenet_v3_small",
                "resumed_from": resumed_from,
                "use_augmentation": use_augmentation,
                "input_size": input_size,
                "epochs": epochs,
                "elapsed_seconds": elapsed_s,
                "train_samples": len(train_rows),
                "val_samples": len(val_rows),
                "test_samples": len(test_rows),
                "challenge_samples": len(challenges),
                "train_groups": train_groups,
                "seen_groups": cumulative_seen_groups,
                "train_sample_ids": train_sample_ids,
                "seen_sample_ids": cumulative_seen_sample_ids,
                "train_image_digests": train_image_digests,
                "seen_image_digests": cumulative_seen_image_digests,
                "train_source_digests": train_source_digests,
                "seen_source_digests": cumulative_seen_source_digests,
                "val_groups": val_groups,
                "test_groups": test_groups,
                "contains_synthetic_bootstrap": any_synthetic,
                "evaluation_kind": evaluation_kind,
                "independent_holdout_verified": independent_holdout_verified,
                "val_accuracy": val_final["accuracy"],
                "test_accuracy": test_res["accuracy"],
                "confusion_matrix": test_res["confusion_matrix"],
                "complete_false_accept_total": test_res["complete_false_accept_total"],
                "onnx_max_abs_diff": round(max_abs_diff, 7) if math.isfinite(max_abs_diff) else None,
                "onnx_parity_ok": parity_ok,
                "onnx_parity_passed": parity_ok,
                "artifact_verified": parity_ok,
                "onnx_path": None,
                "onnx_sha256": None,
                "weights_sha256": pth_sha,
                "manifest_path": str(manifest_csv),
                "exported": False,
                "activated": False,
                "activation_status": "blocked_parity_failed",
                "activation_error": None,
                "completed_at_utc": _utc_now(),
            }

            metrics_path = run_dir / "metrics.json"
            if not parity_ok:
                # Isolate parity-failed ONNX artifact so verified startup/activation never selects it (BE-28)
                failed_onnx_path = run_dir / "label_classifier.failed_parity.onnx"
                staged_onnx_path.replace(failed_onnx_path)
                staged_onnx_path = None
                run_summary["failed_onnx_path"] = str(failed_onnx_path)
                run_summary["failed_onnx_sha256"] = _sha256_file(failed_onnx_path)
                err_msg = (
                    f"ONNX 数值一致性校验未通过 (max_abs_diff={max_abs_diff:.6g} >= {ONNX_PARITY_MAX_ABS_DIFF:g})，"
                    "已隔离异常产物并阻止部署"
                )
                run_summary["activation_error"] = err_msg
                _write_json_atomic(metrics_path, run_summary)
                with self._lock:
                    self._status.update(
                        {
                            "state": "FAILED",
                            "message": err_msg,
                            "error": err_msg,
                            "progress_pct": 100,
                            "exported": False,
                            "activated": False,
                            "activation_status": "blocked_parity_failed",
                            "latest_run": run_summary,
                            "updated_at_utc": _utc_now(),
                        }
                    )
                return

            # Parity passed: promote staged ONNX to verified artifact path
            staged_onnx_path.replace(onnx_path)
            staged_onnx_path = None
            onnx_sha = _sha256_file(onnx_path)
            run_summary["onnx_path"] = str(onnx_path)
            run_summary["onnx_sha256"] = onnx_sha
            run_summary["exported"] = True
            run_summary["activation_status"] = "pending" if on_complete is not None else "exported_only"
            _write_json_atomic(metrics_path, run_summary)

            if on_complete is None:
                with self._lock:
                    self._status.update(
                        {
                            "state": "COMPLETED",
                            "message": f"训练与 ONNX 导出完成（测试集准确率 {test_res['accuracy'] * 100:.1f}%，已导出待激活，未热加载）",
                            "progress_pct": 100,
                            "exported": True,
                            "activated": False,
                            "activation_status": "exported_only",
                            "latest_run": run_summary,
                            "error": None,
                            "updated_at_utc": _utc_now(),
                        }
                    )
                return

            # Keep run in progress while invoking deployment/hot-load callback (BE-29)
            with self._lock:
                self._status.update(
                    {
                        "state": "EXPORTING_ONNX",
                        "message": "ONNX 校验通过，正在注册模型并热加载至实时相机引擎…",
                        "progress_pct": 95,
                        "exported": True,
                        "activated": False,
                        "activation_status": "pending",
                        "updated_at_utc": _utc_now(),
                    }
                )

            activated = False
            activation_error: str | None = None
            try:
                cb_result = on_complete(dict(run_summary))
                if cb_result is False:
                    activation_error = "部署回调返回 False（实时引擎未切换到新模型）"
                elif isinstance(cb_result, dict) and (
                    cb_result.get("activated") is False or cb_result.get("ok") is False
                ):
                    activation_error = str(cb_result.get("error") or "部署回调报告模型激活失败")
                else:
                    activated = True
            except Exception as cb_exc:
                logger.warning("Model activation callback failed for %s: %s", run_id, cb_exc, exc_info=True)
                activation_error = str(cb_exc) or cb_exc.__class__.__name__

            run_summary["activated"] = activated
            run_summary["activation_status"] = "activated" if activated else "activation_failed"
            run_summary["activation_error"] = activation_error
            metrics_write_err = self._persist_metrics_with_retry(metrics_path, run_summary)
            run_summary["metrics_persisted"] = metrics_write_err is None
            run_summary["metrics_write_error"] = metrics_write_err

            if activated:
                if metrics_write_err is None:
                    done_msg = f"训练与 ONNX 导出完成（测试集准确率 {test_res['accuracy'] * 100:.1f}%，已热加载至实时相机引擎）"
                else:
                    done_msg = (
                        f"训练与 ONNX 导出完成（测试集准确率 {test_res['accuracy'] * 100:.1f}%，"
                        f"已热加载至实时相机引擎；但 metrics.json 写入失败，待重试：{metrics_write_err}）"
                    )
                with self._lock:
                    self._status.update(
                        {
                            "state": "COMPLETED",
                            "message": done_msg,
                            "progress_pct": 100,
                            "exported": True,
                            "activated": True,
                            "activation_status": "activated",
                            "metrics_persisted": metrics_write_err is None,
                            "metrics_write_error": metrics_write_err,
                            "latest_run": run_summary,
                            "error": None,
                            "updated_at_utc": _utc_now(),
                        }
                    )
            else:
                fail_msg = f"ONNX 模型已导出，但热加载激活失败（仍运行上一版模型）：{activation_error}"
                with self._lock:
                    self._status.update(
                        {
                            "state": "FAILED",
                            "message": fail_msg,
                            "error": fail_msg,
                            "progress_pct": 100,
                            "exported": True,
                            "activated": False,
                            "activation_status": "activation_failed",
                            "metrics_persisted": metrics_write_err is None,
                            "metrics_write_error": metrics_write_err,
                            "latest_run": run_summary,
                            "updated_at_utc": _utc_now(),
                        }
                    )

        except Exception as exc:
            with self._lock:
                self._status.update(
                    {
                        "state": "FAILED",
                        "message": f"训练异常终止：{exc}",
                        "error": str(exc),
                        "updated_at_utc": _utc_now(),
                    }
                )
        finally:
            if staged_onnx_path is not None:
                try:
                    staged_onnx_path.unlink(missing_ok=True)
                except OSError as cleanup_exc:
                    logger.debug("Failed to clean up staged ONNX file %s: %s", staged_onnx_path, cleanup_exc)
            if snapshot_dir is not None:
                shutil.rmtree(snapshot_dir, ignore_errors=True)
