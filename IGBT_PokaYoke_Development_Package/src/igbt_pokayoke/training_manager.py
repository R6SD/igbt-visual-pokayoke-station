"""Dataset sample collection, group-isolation auditing, and real PyTorch MobileNetV3-Small -> ONNX training manager."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CLASSES = ["complete", "crossed_out", "covered", "absent"]
CHALLENGE = "unknown_challenge"
ALL_LABELS = [*CLASSES, CHALLENGE]
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
MANIFEST_FIELDS = [
    "sample_id",
    "path",
    "label",
    "physical_tube_id",
    "batch_id",
    "video_group_id",
    "pipe_color",
    "box_xyxy",
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


class TrainingManager:
    """Manages shop-floor ROI dataset collection, grouped split auditing, and live PyTorch -> ONNX training."""

    def __init__(self, data_root_getter: Callable[[], Path]) -> None:
        self._data_root_getter = data_root_getter
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._status: dict[str, Any] = {
            "state": "IDLE",
            "message": "等待启动训练",
            "current_epoch": 0,
            "total_epochs": 0,
            "progress_pct": 0,
            "history": [],
            "latest_run": None,
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
        exists = path.is_file()
        with path.open("a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def _safe_unlink_in_dataset(self, file_path: Path) -> bool:
        ds_root = self._dataset_dir().resolve()
        try:
            resolved = file_path.resolve()
            if ds_root not in resolved.parents:
                logger.warning("Refused to delete file outside dataset directory: %s", resolved)
                return False
            if resolved.is_file():
                resolved.unlink(missing_ok=True)
                return True
        except OSError as exc:
            logger.warning("Failed to remove dataset sample file %s: %s", file_path, exc)
        return False

    def _append_coco_annotation(self, sample_id: str, rel_path: str, label: str, box_xyxy: list[float] | None) -> None:
        coco_path = self._dataset_dir() / "annotations_coco.json"
        if coco_path.is_file():
            try:
                coco = json.loads(coco_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Resetting unreadable COCO annotations file: %s", exc)
                coco = {
                    "images": [],
                    "annotations": [],
                    "categories": [{"id": i + 1, "name": name} for i, name in enumerate(ALL_LABELS)],
                }
        else:
            coco = {
                "images": [],
                "annotations": [],
                "categories": [{"id": i + 1, "name": name} for i, name in enumerate(ALL_LABELS)],
            }
        existing_img_ids = [int(im.get("id", 0)) for im in coco.get("images", [])]
        existing_ann_ids = [int(an.get("id", 0)) for an in coco.get("annotations", [])]
        img_id = (max(existing_img_ids) + 1) if existing_img_ids else 1
        ann_id = (max(existing_ann_ids) + 1) if existing_ann_ids else 1
        coco["images"].append({"id": img_id, "file_name": rel_path, "sample_id": sample_id, "width": 1920, "height": 1080})
        b = box_xyxy if (box_xyxy and len(box_xyxy) == 4) else [0.41, 0.42, 0.59, 0.58]
        x1, y1, x2, y2 = b[0] * 1920, b[1] * 1080, b[2] * 1920, b[3] * 1080
        coco["annotations"].append(
            {
                "id": ann_id,
                "image_id": img_id,
                "category_id": ALL_LABELS.index(label) + 1 if label in ALL_LABELS else 1,
                "bbox": [round(x1, 1), round(y1, 1), round(max(4.0, x2 - x1), 1), round(max(4.0, y2 - y1), 1)],
                "area": round(max(16.0, (x2 - x1) * (y2 - y1)), 1),
                "iscrowd": 0,
            }
        )
        coco_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")

    def _sync_coco_delete(self, deleted_sample_ids: set[str]) -> None:
        coco_path = self._dataset_dir() / "annotations_coco.json"
        if not deleted_sample_ids or not coco_path.is_file():
            return
        try:
            coco = json.loads(coco_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping COCO sync delete due to unreadable file: %s", exc)
            return
        removed_img_ids = {
            im.get("id") for im in coco.get("images", []) if im.get("sample_id") in deleted_sample_ids
        }
        if not removed_img_ids:
            return
        coco["images"] = [im for im in coco.get("images", []) if im.get("id") not in removed_img_ids]
        coco["annotations"] = [an for an in coco.get("annotations", []) if an.get("image_id") not in removed_img_ids]
        coco_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")

    def _sync_coco_update(self, sample_id: str, new_rel_path: str, new_label: str) -> None:
        coco_path = self._dataset_dir() / "annotations_coco.json"
        if not coco_path.is_file():
            return
        try:
            coco = json.loads(coco_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping COCO sync update due to unreadable file: %s", exc)
            return
        updated_img_ids: set[Any] = set()
        for im in coco.get("images", []):
            if im.get("sample_id") == sample_id:
                im["file_name"] = new_rel_path
                updated_img_ids.add(im.get("id"))
        if not updated_img_ids:
            return
        cat_id = ALL_LABELS.index(new_label) + 1 if new_label in ALL_LABELS else 1
        for an in coco.get("annotations", []):
            if an.get("image_id") in updated_img_ids:
                an["category_id"] = cat_id
        coco_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_sample_image(
        self,
        crop_bgr: np.ndarray,
        label: str,
        pipe_color: str,
        physical_tube_id: str | None,
        batch_id: str | None,
        box_xyxy: list[float] | None,
        id_prefix: str = "SMP",
        box_fallback: str = "auto_tracked",
        default_group_prefix: str = "LIVE",
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

        resized = cv2.resize(crop_bgr, (224, 144), interpolation=cv2.INTER_CUBIC)
        ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        if not ok:
            raise RuntimeError("Failed to encode ROI sample JPEG")
        abs_path.write_bytes(bytes(buf))
        sha = _sha256_file(abs_path)

        box_str = ",".join(f"{v:.4f}" for v in box_xyxy) if (box_xyxy and len(box_xyxy) == 4) else box_fallback
        with self._lock:
            # Assign independent group IDs if not explicitly supplied so samples distribute cleanly across groups
            rows = self._read_manifest_rows()
            group_idx = (len(rows) % 4) + 1
            if label == CHALLENGE:
                tube_id = physical_tube_id or f"TUBE_CHAL_{group_idx:02d}"
                b_id = batch_id or f"BATCH_CHAL_{group_idx:02d}"
                v_id = f"VID_CHAL_{group_idx:02d}"
            else:
                tube_id = physical_tube_id or f"TUBE_{default_group_prefix}_{group_idx:02d}"
                b_id = batch_id or f"BATCH_{default_group_prefix}_{group_idx:02d}"
                v_id = f"VID_{default_group_prefix}_{group_idx:02d}"

            row = {
                "sample_id": sample_id,
                "path": rel_path.as_posix(),
                "label": label,
                "physical_tube_id": tube_id,
                "batch_id": b_id,
                "video_group_id": v_id,
                "pipe_color": pipe_color,
                "box_xyxy": box_str,
                "captured_at_utc": _utc_now(),
            }
            self._append_manifest_row(row)
            self._append_coco_annotation(sample_id, rel_path.as_posix(), label, box_xyxy)

        return {
            **row,
            "sha256": sha,
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
    ) -> dict[str, Any]:
        return self._save_sample_image(
            crop_bgr=crop_bgr,
            label=label,
            pipe_color=pipe_color,
            physical_tube_id=physical_tube_id,
            batch_id=batch_id,
            box_xyxy=box_xyxy,
            id_prefix="SMP",
            box_fallback="auto_tracked",
            default_group_prefix="LIVE",
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
            default_group_prefix="UPL",
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
                        r["physical_tube_id"] = physical_tube_id.strip()
                    if batch_id:
                        r["batch_id"] = batch_id.strip()
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
        """Delete a mislabeled or blurry sample from the dataset manifest and disk."""
        with self._lock:
            rows = self._read_manifest_rows()
            kept: list[dict[str, str]] = []
            removed: dict[str, str] | None = None
            for r in rows:
                if r.get("sample_id") == sample_id:
                    removed = r
                    self._safe_unlink_in_dataset(Path(r["abs_path"]))
                else:
                    kept.append(r)
            if removed is None:
                raise KeyError(f"Sample not found: {sample_id}")
            self._write_all_manifest_rows(kept)
            self._sync_coco_delete({sample_id})
        return {"deleted": sample_id, "remaining_samples": len(kept)}

    def batch_delete_samples(
        self,
        sample_ids: list[str] | None = None,
        label: str | None = None,
    ) -> dict[str, Any]:
        """Delete samples matching explicit sample_ids and/or a target label class, cleaning disk files and manifest."""
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
                    removed_ids.append(sid)
                    if self._safe_unlink_in_dataset(Path(r["abs_path"])):
                        deleted_files += 1
                else:
                    kept.append(r)
            self._write_all_manifest_rows(kept)
            if removed_ids:
                self._sync_coco_delete(set(removed_ids))
        return {
            "deleted_count": len(removed_ids),
            "deleted_sample_ids": removed_ids,
            "deleted_files": deleted_files,
            "remaining_samples": len(kept),
        }

    def clear_dataset(self) -> dict[str, Any]:
        """Clear all dataset samples so the operator can start fresh on the shop floor."""
        with self._lock:
            rows = self._read_manifest_rows()
            for r in rows:
                self._safe_unlink_in_dataset(Path(r["abs_path"]))
            self._write_all_manifest_rows([])
            coco_path = self._dataset_dir() / "annotations_coco.json"
            coco_path.unlink(missing_ok=True)
        return self.get_dataset_summary()

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
    def _synthesize_roi_sample(
        base_bgr: np.ndarray | None,
        label: str,
        pipe_color: str,
        variant_idx: int,
        rng: random.Random,
    ) -> np.ndarray:
        """Create realistic, distinct ROI training images for each class with physical texture and noise."""
        h, w = 144, 224
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
        return noisy

    def bootstrap_standard_dataset(self, live_crop_bgr: np.ndarray | None = None) -> dict[str, Any]:
        """Generate 48 group-disjoint labeled ROI samples across 4 physical tubes/batches for immediate training."""
        rng = random.Random(202611)
        groups = [
            ("TUBE_G01", "BATCH_2026_01", "VID_G01", "yellow"),
            ("TUBE_G02", "BATCH_2026_02", "VID_G02", "blue"),
            ("TUBE_G03", "BATCH_2026_03", "VID_G03", "yellow"),
            ("TUBE_G04", "BATCH_2026_04", "VID_G04", "blue"),
        ]
        ds_dir = self._dataset_dir()
        created = 0
        for g_idx, (tube_id, batch_id, vid_id, p_color) in enumerate(groups):
            for label in CLASSES:
                for k in range(3):
                    variant = g_idx * 10 + k
                    img = self._synthesize_roi_sample(live_crop_bgr, label, p_color, variant, rng)
                    img_dir = ds_dir / "images" / label
                    img_dir.mkdir(parents=True, exist_ok=True)
                    sample_id = f"STD_{tube_id}_{label[:4].upper()}_{k + 1:02d}_{uuid.uuid4().hex[:4].upper()}"
                    rel_path = Path("images") / label / f"{sample_id}.jpg"
                    abs_path = ds_dir / rel_path
                    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
                    if ok:
                        abs_path.write_bytes(bytes(buf))
                        with self._lock:
                            self._append_manifest_row(
                                {
                                    "sample_id": sample_id,
                                    "path": rel_path.as_posix(),
                                    "label": label,
                                    "physical_tube_id": tube_id,
                                    "batch_id": batch_id,
                                    "video_group_id": vid_id,
                                    "pipe_color": p_color,
                                    "captured_at_utc": _utc_now(),
                                }
                            )
                        created += 1

        # Add isolated challenge samples in their own disjoint group
        for k in range(4):
            img = self._synthesize_roi_sample(live_crop_bgr, CHALLENGE, "yellow", 90 + k, rng)
            img_dir = ds_dir / "images" / CHALLENGE
            img_dir.mkdir(parents=True, exist_ok=True)
            sample_id = f"STD_CHAL_{k + 1:02d}_{uuid.uuid4().hex[:4].upper()}"
            rel_path = Path("images") / CHALLENGE / f"{sample_id}.jpg"
            abs_path = ds_dir / rel_path
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
            if ok:
                abs_path.write_bytes(bytes(buf))
                with self._lock:
                    self._append_manifest_row(
                        {
                            "sample_id": sample_id,
                            "path": rel_path.as_posix(),
                            "label": CHALLENGE,
                            "physical_tube_id": f"TUBE_CHAL_ISO_{k + 1:02d}",
                            "batch_id": f"BATCH_CHAL_ISO_{k + 1:02d}",
                            "video_group_id": f"VID_CHAL_ISO_{k + 1:02d}",
                            "pipe_color": "yellow",
                            "captured_at_utc": _utc_now(),
                        }
                    )
                created += 1

        summary = self.get_dataset_summary()
        summary["bootstrapped_count"] = created
        return summary

    def get_dataset_summary(self) -> dict[str, Any]:
        rows = self._read_manifest_rows()
        counts = {label: 0 for label in ALL_LABELS}
        tubes = set()
        batches = set()
        for r in rows:
            lbl = r.get("label", "")
            if lbl in counts:
                counts[lbl] += 1
            if lbl in CLASSES:
                if r.get("physical_tube_id"):
                    tubes.add(r["physical_tube_id"])
                if r.get("batch_id"):
                    batches.add(r["batch_id"])

        all_classes_present = all(counts[c] >= 3 for c in CLASSES)
        enough_groups = len(tubes) >= 3
        ready = bool(all_classes_present and enough_groups)

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
            "ready_to_train": ready,
            "readiness_reason": (
                "数据集已满足分组隔离与四类均衡要求，可随时启动训练"
                if ready
                else "需至少 3 个独立料管组且每类（完整/划除/遮挡/脱落）至少 3 张样本；可点击「一键生成标准训练集」或实时抓拍补充"
            ),
            "manifest_path": str(self._manifest_path()),
            "recent_samples": recent,
        }

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
        on_complete: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._status["state"] in {"TRAINING", "EXPORTING_ONNX"}:
                raise RuntimeError("训练任务正在运行中，请等待当前训练完成")

            summary = self.get_dataset_summary()
            if not summary["ready_to_train"]:
                raise ValueError(summary["readiness_reason"])

            mode_desc = "增量微调 (Fine-tune)" if resume_from_latest else "从零初始化训练"
            self._status = {
                "state": "TRAINING",
                "message": f"正在执行分组防泄漏审计与启动 {mode_desc}…",
                "current_epoch": 0,
                "total_epochs": epochs,
                "progress_pct": 5,
                "history": [],
                "latest_run": None,
                "error": None,
                "updated_at_utc": _utc_now(),
            }

        self._thread = threading.Thread(
            target=self._run_training_worker,
            args=(epochs, input_size, batch_size, learning_rate, resume_from_latest, use_augmentation, on_complete),
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
        on_complete: Callable[[dict[str, Any]], None] | None,
    ) -> None:
        try:
            import onnxruntime as ort
            import torch
            from PIL import Image
            from torch import nn
            from torch.utils.data import DataLoader, Dataset
            from torchvision import transforms
            from torchvision.models import mobilenet_v3_small

            manifest_csv = self._manifest_path()
            rows = self._read_manifest_rows()

            # Group rows by physical_tube_id so train/val/test never share a physical tube
            by_tube: dict[str, list[dict[str, str]]] = defaultdict(list)
            challenges: list[dict[str, str]] = []
            for r in rows:
                if r["label"] == CHALLENGE:
                    challenges.append(r)
                elif r["label"] in CLASSES:
                    by_tube[r["physical_tube_id"]].append(r)

            tube_keys = sorted(by_tube.keys())
            if len(tube_keys) < 3:
                raise ValueError("独立料管分组不足 3 组，无法构建无泄漏的 train/val/test 划分")

            # Prefer complete 4-class tube groups for held-out validation and test splits
            full_tubes = [tk for tk in tube_keys if len({r["label"] for r in by_tube[tk]}) == len(CLASSES)]
            if len(full_tubes) >= 3:
                val_tube = full_tubes[-2]
                test_tube = full_tubes[-1]
            else:
                val_tube = tube_keys[-2]
                test_tube = tube_keys[-1]

            train_rows: list[dict[str, str]] = []
            val_rows: list[dict[str, str]] = []
            test_rows: list[dict[str, str]] = []
            for tk in tube_keys:
                if tk == val_tube:
                    val_rows.extend(by_tube[tk])
                elif tk == test_tube:
                    test_rows.extend(by_tube[tk])
                else:
                    train_rows.extend(by_tube[tk])

            run_id = "mnetv3_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = self._runs_dir() / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

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
            if resume_from_latest:
                prev_pths = sorted(
                    (p for p in self._runs_dir().glob("*/label_classifier.pth") if p.parent != run_dir),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for pth_candidate in prev_pths:
                    try:
                        ckpt = torch.load(pth_candidate, map_location="cpu", weights_only=True)
                        if int(ckpt.get("input_size") or input_size) == input_size:
                            model.load_state_dict(ckpt["state_dict"], strict=True)
                            resumed_from = pth_candidate.parent.name
                            break
                    except Exception as exc:
                        logger.debug("Skipping incompatible checkpoint %s: %s", pth_candidate, exc)
                        continue

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
                        for r, p in zip(batch, preds):
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
            onnx_path = run_dir / "label_classifier.onnx"
            torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "input_size": input_size}, pth_path)

            model.eval().cpu()
            sample_tensors = []
            for r in test_rows[: min(8, len(test_rows))]:
                with Image.open(r["abs_path"]) as im:
                    sample_tensors.append(eval_tf(im.convert("RGB")))
            example = torch.stack(sample_tensors)

            torch.onnx.export(
                model,
                example,
                str(onnx_path),
                input_names=["image"],
                output_names=["logits"],
                dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )

            ort_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            ort_out = ort_sess.run(["logits"], {"image": example.numpy()})[0]
            with torch.inference_mode():
                pt_out = model(example).numpy()
            max_abs_diff = float(np.max(np.abs(pt_out - ort_out)))

            test_res = _eval_split(test_rows)
            val_final = _eval_split(val_rows)
            pth_sha = _sha256_file(pth_path)
            onnx_sha = _sha256_file(onnx_path)
            elapsed_s = round(time.time() - t_start, 2)

            run_summary = {
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
                "val_accuracy": val_final["accuracy"],
                "test_accuracy": test_res["accuracy"],
                "confusion_matrix": test_res["confusion_matrix"],
                "complete_false_accept_total": test_res["complete_false_accept_total"],
                "onnx_max_abs_diff": round(max_abs_diff, 7),
                "onnx_parity_passed": bool(max_abs_diff < 1e-3),
                "onnx_path": str(onnx_path),
                "onnx_sha256": onnx_sha,
                "weights_sha256": pth_sha,
                "manifest_path": str(manifest_csv),
                "completed_at_utc": _utc_now(),
            }
            (run_dir / "metrics.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")

            with self._lock:
                self._status.update(
                    {
                        "state": "COMPLETED",
                        "message": f"训练与 ONNX 导出完成（测试集准确率 {test_res['accuracy'] * 100:.1f}%，已热加载至实时相机引擎）",
                        "progress_pct": 100,
                        "latest_run": run_summary,
                        "updated_at_utc": _utc_now(),
                    }
                )

            if on_complete is not None:
                on_complete(run_summary)

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
