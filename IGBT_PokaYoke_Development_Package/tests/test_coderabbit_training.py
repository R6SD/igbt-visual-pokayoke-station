"""Comprehensive unit and integration tests for Wave 3 CodeRabbit training/dataset hardening (BE-20..BE-29, BE-53)."""
from __future__ import annotations

import csv
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from igbt_pokayoke import training_manager as tm_mod
from igbt_pokayoke.training_manager import (
    ALL_LABELS,
    CLASSES,
    SAVED_IMAGE_HEIGHT,
    SAVED_IMAGE_WIDTH,
    UNGROUPED_TUBE_ID,
    TrainingManager,
)


def _make_dummy_crop(seed: int = 0, color_bgr: tuple[int, int, int] = (60, 175, 210)) -> np.ndarray:
    rng = np.random.RandomState(seed)
    img = np.full((120, 180, 3), color_bgr, dtype=np.uint8)
    cv2.rectangle(img, (30, 20), (150, 100), (235, 238, 240), -1)
    noise = rng.randint(-6, 7, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _encode_jpg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("Failed to encode test JPEG")
    return bytes(buf)


def _populate_real_tubes(tm: TrainingManager, tube_ids: list[str], samples_per_class: int = 2) -> None:
    seed = 100
    for t_idx, tube_id in enumerate(tube_ids):
        pipe_color = "yellow" if t_idx % 2 == 0 else "blue"
        for c_idx, label in enumerate(CLASSES):
            for _ in range(samples_per_class):
                seed += 1
                base_color = (40 + c_idx * 45, 180 - c_idx * 30, 210 - t_idx * 20)
                tm.capture_sample(
                    crop_bgr=_make_dummy_crop(seed=seed, color_bgr=base_color),
                    label=label,
                    pipe_color=pipe_color,
                    physical_tube_id=tube_id,
                    batch_id=f"BATCH_{tube_id}",
                    box_xyxy=[0.2, 0.25, 0.8, 0.75],
                )


def _wait_for_training(tm: TrainingManager, timeout: float = 90.0) -> dict:
    if tm._thread is not None:
        tm._thread.join(timeout=timeout)
        if tm._thread.is_alive():
            raise TimeoutError("Training worker thread did not finish in time")
    return tm.get_training_status()


class Wave3TrainingDataHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.data_root = Path(self._tmp.name)
        self._managers: list[TrainingManager] = []
        self.tm = self._make_manager(self.data_root)

    def _make_manager(self, root: Path) -> TrainingManager:
        mgr = TrainingManager(data_root_getter=lambda r=root: r)
        self._managers.append(mgr)
        return mgr

    def tearDown(self) -> None:
        still_alive: list[str] = []
        for idx, mgr in enumerate(self._managers):
            worker = mgr._thread
            if worker is not None and worker.is_alive():
                worker.join(timeout=30.0)
                if worker.is_alive():
                    still_alive.append(f"manager[{idx}]:{worker.name}")
        if still_alive:
            # Do not delete temp files while any worker thread could still be reading/writing them (R2-42)
            raise AssertionError(f"Training worker thread(s) did not exit before teardown: {', '.join(still_alive)}")
        self._tmp.cleanup()

    # ------------------------------------------------------------------
    # BE-20: Do not replace existing COCO annotations after a read failure
    # ------------------------------------------------------------------
    def test_be20_coco_read_failure_preserves_existing_file_and_supports_rebuild(self) -> None:
        self.tm.capture_sample(_make_dummy_crop(1), "complete", physical_tube_id="TUBE_01")
        self.tm.capture_sample(_make_dummy_crop(2), "crossed_out", physical_tube_id="TUBE_01")
        coco_path = self.data_root / "dataset" / "annotations_coco.json"
        original_text = coco_path.read_text(encoding="utf-8")
        original_doc = json.loads(original_text)
        self.assertEqual(len(original_doc["images"]), 2)
        self.assertEqual(len(original_doc["annotations"]), 2)

        # Transient OSError on reading annotations_coco.json must preserve original file untouched
        real_read_text = Path.read_text

        def flaky_read_text(path_obj: Path, *args, **kwargs):
            if path_obj.name == "annotations_coco.json":
                raise OSError("Simulated transient read failure")
            return real_read_text(path_obj, *args, **kwargs)

        with mock.patch.object(Path, "read_text", autospec=True, side_effect=flaky_read_text):
            sample3 = self.tm.capture_sample(_make_dummy_crop(3), "covered", physical_tube_id="TUBE_02")

        self.assertFalse(sample3["coco_synced"])
        self.assertEqual(coco_path.read_text(encoding="utf-8"), original_text)

        # Corrupted JSON on disk must also be preserved (never overwritten with a 1-sample skeleton)
        corrupt_text = "{invalid-json-truncated"
        coco_path.write_text(corrupt_text, encoding="utf-8")
        sample4 = self.tm.capture_sample(_make_dummy_crop(4), "absent", physical_tube_id="TUBE_02")
        self.assertFalse(sample4["coco_synced"])
        self.assertEqual(coco_path.read_text(encoding="utf-8"), corrupt_text)

        # Authoritative rebuild from manifest CSV restores all 4 samples with actual saved dimensions
        rebuilt = self.tm.rebuild_coco_from_manifest()
        self.assertEqual(rebuilt["rebuilt_annotations"], 4)
        rebuilt_doc = json.loads(coco_path.read_text(encoding="utf-8"))
        self.assertEqual(len(rebuilt_doc["images"]), 4)
        self.assertEqual(len(rebuilt_doc["annotations"]), 4)
        for im in rebuilt_doc["images"]:
            self.assertEqual(im["width"], SAVED_IMAGE_WIDTH)
            self.assertEqual(im["height"], SAVED_IMAGE_HEIGHT)

    # ------------------------------------------------------------------
    # BE-21: Write COCO coordinates for the saved image (224x144 crop space)
    # ------------------------------------------------------------------
    def test_be21_coco_records_saved_dimensions_and_crop_coordinates(self) -> None:
        # 1. Custom full-camera-frame capture (crop IS that box -> [0, 0, 224, 144])
        s_custom = self.tm.capture_sample(
            crop_bgr=_make_dummy_crop(10),
            label="complete",
            physical_tube_id="TUBE_C1",
            box_xyxy=[0.20, 0.30, 0.60, 0.70],
        )
        # 2. Sub-box inside a wider crop bounds -> projected into crop space
        s_sub = self.tm.capture_sample(
            crop_bgr=_make_dummy_crop(11),
            label="crossed_out",
            physical_tube_id="TUBE_C1",
            box_xyxy=[0.30, 0.35, 0.50, 0.65],
            crop_bounds_xyxy=[0.20, 0.20, 0.60, 0.80],
        )
        # 3. Auto-tracked capture with missing box -> fallback covers saved crop
        s_missing = self.tm.capture_sample(
            crop_bgr=_make_dummy_crop(12),
            label="covered",
            physical_tube_id="TUBE_C1",
            box_xyxy=None,
        )
        # 4. Uploaded bytes
        s_upload = self.tm.upload_sample_bytes(
            image_bytes=_encode_jpg(_make_dummy_crop(13)),
            label="absent",
            physical_tube_id="TUBE_C1",
        )

        coco_path = self.data_root / "dataset" / "annotations_coco.json"
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
        img_by_sid = {im["sample_id"]: im for im in coco["images"]}
        ann_by_img_id = {an["image_id"]: an for an in coco["annotations"]}

        for sid in (s_custom["sample_id"], s_sub["sample_id"], s_missing["sample_id"], s_upload["sample_id"]):
            im = img_by_sid[sid]
            self.assertEqual(im["width"], SAVED_IMAGE_WIDTH)
            self.assertEqual(im["height"], SAVED_IMAGE_HEIGHT)
            self.assertNotEqual(im["width"], 1920)
            self.assertNotEqual(im["height"], 1080)

        custom_ann = ann_by_img_id[img_by_sid[s_custom["sample_id"]]["id"]]
        self.assertEqual(custom_ann["bbox"], [0.0, 0.0, 224.0, 144.0])
        self.assertEqual(custom_ann["area"], 32256.0)
        self.assertEqual(s_custom["box_xyxy"], "0.2000,0.3000,0.6000,0.7000")

        sub_ann = ann_by_img_id[img_by_sid[s_sub["sample_id"]]["id"]]
        self.assertEqual(sub_ann["bbox"], [56.0, 36.0, 112.0, 72.0])
        self.assertEqual(sub_ann["area"], 8064.0)

        missing_ann = ann_by_img_id[img_by_sid[s_missing["sample_id"]]["id"]]
        self.assertEqual(missing_ann["bbox"], [0.0, 0.0, 224.0, 144.0])

        upload_ann = ann_by_img_id[img_by_sid[s_upload["sample_id"]]["id"]]
        self.assertEqual(upload_ann["bbox"], [0.0, 0.0, 224.0, 144.0])

    # ------------------------------------------------------------------
    # BE-22: Do not assign independent tube IDs to ungrouped captures
    # ------------------------------------------------------------------
    def test_be22_ungrouped_captures_use_shared_bucket_and_cannot_fake_holdouts(self) -> None:
        tube_ids_assigned = set()
        for idx in range(16):
            lbl = CLASSES[idx % len(CLASSES)]
            sample = self.tm.capture_sample(
                crop_bgr=_make_dummy_crop(200 + idx),
                label=lbl,
                physical_tube_id=None,
                batch_id=None,
            )
            tube_ids_assigned.add(sample["physical_tube_id"])

        self.assertEqual(tube_ids_assigned, {UNGROUPED_TUBE_ID})

        summary_only_ungrouped = self.tm.get_dataset_summary()
        self.assertEqual(summary_only_ungrouped["independent_tube_groups"], 0)
        self.assertEqual(summary_only_ungrouped["ungrouped_samples"], 16)
        self.assertFalse(summary_only_ungrouped["ready_to_train"])
        with self.assertRaises(ValueError):
            self.tm.start_training(epochs=2)

        # Adding only 2 real independent tubes alongside ungrouped captures still cannot fake 3 groups
        _populate_real_tubes(self.tm, ["TUBE_REAL_01", "TUBE_REAL_02"], samples_per_class=2)
        summary_two = self.tm.get_dataset_summary()
        self.assertEqual(summary_two["independent_tube_groups"], 2)
        self.assertFalse(summary_two["ready_to_train"])

        # Adding a 3rd real independent tube unlocks training, while UNGROUPED is strictly excluded from holdouts
        _populate_real_tubes(self.tm, ["TUBE_REAL_03"], samples_per_class=2)
        summary_three = self.tm.get_dataset_summary()
        self.assertEqual(summary_three["independent_tube_groups"], 3)
        self.assertTrue(summary_three["ready_to_train"])

        rows = self.tm._read_manifest_rows()
        split = self.tm._select_group_split(rows)
        self.assertNotIn(UNGROUPED_TUBE_ID, split["val_groups"])
        self.assertNotIn(UNGROUPED_TUBE_ID, split["test_groups"])
        self.assertEqual(split["ungrouped_count"], 16)
        for row in (*split["val_rows"], *split["test_rows"]):
            self.assertNotEqual(row["physical_tube_id"], UNGROUPED_TUBE_ID)

    # ------------------------------------------------------------------
    # BE-23: Keep failed file deletions retryable
    # ------------------------------------------------------------------
    def test_be23_failed_unlink_retains_manifest_row_and_allows_retry(self) -> None:
        s1 = self.tm.capture_sample(_make_dummy_crop(301), "complete", physical_tube_id="TUBE_D1")
        s2 = self.tm.capture_sample(_make_dummy_crop(302), "crossed_out", physical_tube_id="TUBE_D1")
        s3 = self.tm.capture_sample(_make_dummy_crop(303), "covered", physical_tube_id="TUBE_D1")
        s1_path = self.tm.get_sample_image_path(s1["sample_id"])
        self.assertIsNotNone(s1_path)
        assert s1_path is not None

        real_unlink = Path.unlink
        locked_paths = {s1_path.resolve()}

        def locked_unlink(path_obj: Path, *args, **kwargs):
            if path_obj.resolve() in locked_paths:
                raise PermissionError("Simulated locked file")
            return real_unlink(path_obj, *args, **kwargs)

        coco_path = self.data_root / "dataset" / "annotations_coco.json"

        with mock.patch.object(Path, "unlink", autospec=True, side_effect=locked_unlink):
            # Single delete fails -> raises OSError and retains manifest + COCO + file
            with self.assertRaises(OSError):
                self.tm.delete_sample(s1["sample_id"])
            self.assertEqual(len(self.tm.list_samples()), 3)
            self.assertTrue(s1_path.is_file())
            self.assertEqual(len(json.loads(coco_path.read_text(encoding="utf-8"))["images"]), 3)

            # Batch delete deletes s2 but retains locked s1 and reports it in failed_deletions
            batch_res = self.tm.batch_delete_samples(sample_ids=[s1["sample_id"], s2["sample_id"]])
            self.assertEqual(batch_res["deleted_count"], 1)
            self.assertEqual(batch_res["deleted_sample_ids"], [s2["sample_id"]])
            self.assertEqual(batch_res["failed_count"], 1)
            self.assertEqual(batch_res["failed_deletions"][0]["sample_id"], s1["sample_id"])
            self.assertEqual(batch_res["remaining_samples"], 2)
            self.assertEqual(
                {item["sample_id"] for item in self.tm.list_samples()},
                {s1["sample_id"], s3["sample_id"]},
            )

            # Clear deletes s3 but retains locked s1 and preserves its COCO annotation
            clear_res = self.tm.clear_dataset()
            self.assertFalse(clear_res["cleared"])
            self.assertEqual(clear_res["failed_count"], 1)
            self.assertEqual(clear_res["total_samples"], 1)
            self.assertEqual(self.tm.list_samples()[0]["sample_id"], s1["sample_id"])
            remaining_coco = json.loads(coco_path.read_text(encoding="utf-8"))
            self.assertEqual([im["sample_id"] for im in remaining_coco["images"]], [s1["sample_id"]])

        # Once unlocked, retry deletes s1 cleanly
        retry_clear = self.tm.clear_dataset()
        self.assertTrue(retry_clear["cleared"])
        self.assertEqual(retry_clear["failed_count"], 0)
        self.assertEqual(retry_clear["total_samples"], 0)
        self.assertFalse(s1_path.exists())
        self.assertFalse(coco_path.exists())

        # ENOENT (file vanished between read and unlink) cleanly removes manifest row without error (R2-43)
        s4 = self.tm.capture_sample(_make_dummy_crop(304), "absent", physical_tube_id="TUBE_D2")
        s4_path = self.tm.get_sample_image_path(s4["sample_id"])
        assert s4_path is not None

        def enoent_unlink(path_obj: Path, *args, **kwargs):
            if path_obj.resolve() == s4_path.resolve():
                real_unlink(path_obj)
                raise FileNotFoundError("Already removed externally")
            return real_unlink(path_obj, *args, **kwargs)

        with mock.patch.object(Path, "unlink", autospec=True, side_effect=enoent_unlink):
            del_enoent = self.tm.delete_sample(s4["sample_id"])
        self.assertEqual(del_enoent["deleted"], s4["sample_id"])
        self.assertFalse(del_enoent["deleted_file"])
        self.assertEqual(del_enoent["remaining_samples"], 0)
        self.assertFalse(s4_path.exists())
        self.assertEqual(self.tm.list_samples(), [])

    # ------------------------------------------------------------------
    # BE-24 & BE-53: Bootstrap source-lineage isolation + COCO export
    # ------------------------------------------------------------------
    def test_be24_and_be53_bootstrap_isolates_live_crop_and_writes_coco(self) -> None:
        # Distinctive magenta live crop so any blending into holdout groups changes mean B/R channel dramatically
        live_crop = np.full((144, 224, 3), (250, 0, 250), dtype=np.uint8)
        summary = self.tm.bootstrap_standard_dataset(live_crop_bgr=live_crop)

        self.assertEqual(summary["bootstrapped_count"], 52)
        self.assertTrue(summary["coco_synced"])
        self.assertTrue(summary["is_synthetic_bootstrap"])
        self.assertEqual(summary["dataset_origin"], "synthetic_smoke_test")
        # Single live crop must ONLY be assigned to ONE group (never leaked into holdouts!)
        self.assertEqual(summary["live_crop_groups"], ["TUBE_G01"])

        # Compare generated pixels against a no-live-crop baseline to prove TUBE_G02..G04 have zero live crop leakage
        baseline_root = self.data_root / "baseline_compare"
        baseline_tm = self._make_manager(baseline_root)
        baseline_tm.bootstrap_standard_dataset(live_crop_bgr=None)

        live_rows = self.tm._read_manifest_rows()
        base_rows = baseline_tm._read_manifest_rows()
        self.assertEqual(len(live_rows), 52)
        self.assertEqual(len(base_rows), 52)

        for lr, br in zip(live_rows, base_rows, strict=True):
            l_img = cv2.imread(lr["abs_path"], cv2.IMREAD_COLOR)
            b_img = cv2.imread(br["abs_path"], cv2.IMREAD_COLOR)
            self.assertIsNotNone(l_img)
            self.assertIsNotNone(b_img)
            diff = float(np.mean(np.abs(l_img.astype(np.float32) - b_img.astype(np.float32))))
            if lr["physical_tube_id"] == "TUBE_G01":
                self.assertGreater(diff, 5.0, "TUBE_G01 should reflect the blended live crop")
            else:
                self.assertLess(
                    diff,
                    0.01,
                    f"Group {lr['physical_tube_id']} ({lr['label']}) leaked pixels from single live_crop_bgr!",
                )

        # BE-53 / R2-44: All 52 bootstrap samples must appear in annotations_coco.json with positive crop bboxes
        coco_path = self.data_root / "dataset" / "annotations_coco.json"
        self.assertTrue(coco_path.is_file())
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
        self.assertEqual(len(coco["images"]), 52)
        self.assertEqual(len(coco["annotations"]), 52)
        self.assertEqual(self.tm.storage_stats()["coco_annotations"], 52)
        for im, an in zip(coco["images"], coco["annotations"], strict=True):
            self.assertEqual(im["width"], SAVED_IMAGE_WIDTH)
            self.assertEqual(im["height"], SAVED_IMAGE_HEIGHT)
            bx, by, bw, bh = an["bbox"]
            self.assertGreater(bw, 0.0)
            self.assertGreater(bh, 0.0)
            self.assertGreaterEqual(bx, 0.0)
            self.assertGreaterEqual(by, 0.0)
            self.assertLessEqual(bx + bw, SAVED_IMAGE_WIDTH + 0.1)
            self.assertLessEqual(by + bh, SAVED_IMAGE_HEIGHT + 0.1)
            self.assertIn(an["category_id"], range(1, len(ALL_LABELS) + 1))

        # Duplicate crops passed via live_crops are deduplicated so identical images never span multiple groups
        dup_root = self.data_root / "dup_crops_test"
        dup_tm = self._make_manager(dup_root)
        dup_summary = dup_tm.bootstrap_standard_dataset(live_crops=[live_crop, live_crop.copy(), live_crop.copy()])
        self.assertEqual(dup_summary["live_crop_groups"], ["TUBE_G01"])

        # Training on synthetic bootstrap data must report evaluation_kind="synthetic_smoke_test"
        self.tm.start_training(epochs=2, input_size=64, batch_size=8, resume_from_latest=False, use_augmentation=False)
        boot_train_status = _wait_for_training(self.tm)
        self.assertEqual(boot_train_status["state"], "COMPLETED", boot_train_status)
        boot_run = boot_train_status["latest_run"]
        self.assertEqual(boot_run["evaluation_kind"], "synthetic_smoke_test")
        self.assertTrue(boot_run["contains_synthetic_bootstrap"])
        self.assertFalse(boot_run["independent_holdout_verified"])

    # ------------------------------------------------------------------
    # BE-26: Check class coverage after choosing the split
    # ------------------------------------------------------------------
    def test_be26_post_split_class_coverage_rejects_degenerate_and_solves_asymmetric(self) -> None:
        # Degenerate layout: 3 tubes, >=3 samples/class, but no holdout choice leaves all 4 classes in train
        for i in range(3):
            self.tm.capture_sample(_make_dummy_crop(400 + i), "complete", physical_tube_id="TUBE_A")
            self.tm.capture_sample(_make_dummy_crop(410 + i), "crossed_out", physical_tube_id="TUBE_B")
            self.tm.capture_sample(_make_dummy_crop(420 + i), "covered", physical_tube_id="TUBE_C")
            self.tm.capture_sample(_make_dummy_crop(430 + i), "absent", physical_tube_id="TUBE_C")

        summary = self.tm.get_dataset_summary()
        self.assertEqual(summary["independent_tube_groups"], 3)
        self.assertFalse(summary["ready_to_train"])
        for cls in CLASSES:
            self.assertIn(cls, summary["readiness_reason"])
        with self.assertRaises(ValueError) as ctx:
            self.tm.start_training(epochs=2)
        for cls in CLASSES:
            self.assertIn(cls, str(ctx.exception))

        # Salvageable asymmetric layout: TUBE_A covers all 4 classes, TUBE_B & TUBE_C hold partial classes
        asym_root = self.data_root / "asym_test"
        asym_tm = self._make_manager(asym_root)
        for i in range(3):
            for cls in CLASSES:
                asym_tm.capture_sample(_make_dummy_crop(500 + i * 10 + CLASSES.index(cls)), cls, physical_tube_id="TUBE_A")
        asym_tm.capture_sample(_make_dummy_crop(550), "complete", physical_tube_id="TUBE_B")
        asym_tm.capture_sample(_make_dummy_crop(551), "crossed_out", physical_tube_id="TUBE_B")
        asym_tm.capture_sample(_make_dummy_crop(552), "covered", physical_tube_id="TUBE_C")
        asym_tm.capture_sample(_make_dummy_crop(553), "absent", physical_tube_id="TUBE_C")

        asym_summary = asym_tm.get_dataset_summary()
        self.assertTrue(asym_summary["ready_to_train"])
        split = asym_tm._select_group_split(asym_tm._read_manifest_rows())
        self.assertEqual(split["train_groups"], ["TUBE_A"])
        self.assertEqual({r["label"] for r in split["train_rows"]}, set(CLASSES))

    # ------------------------------------------------------------------
    # BE-25 / R2-45: Synchronized immutable run-scoped snapshot survives concurrent relabel/delete/clear
    # ------------------------------------------------------------------
    def test_be25_training_survives_concurrent_relabel_delete_and_clear(self) -> None:
        from PIL import Image

        _populate_real_tubes(self.tm, ["TUBE_S1", "TUBE_S2", "TUBE_S3"], samples_per_class=2)
        samples_before = self.tm.list_samples()
        self.assertEqual(len(samples_before), 24)
        original_paths = [self.tm.get_sample_image_path(s["sample_id"]) for s in samples_before]
        self.assertTrue(all(p is not None and p.is_file() for p in original_paths))

        worker_paused = threading.Event()
        release_worker = threading.Event()
        real_open = Image.open
        opened_in_worker = False

        def synchronized_open(fp, *args, **kwargs):
            nonlocal opened_in_worker
            if threading.current_thread().name == "MobileNetV3Trainer" and not opened_in_worker:
                opened_in_worker = True
                worker_paused.set()
                release_worker.wait(timeout=30.0)
            return real_open(fp, *args, **kwargs)

        with mock.patch.object(Image, "open", side_effect=synchronized_open):
            try:
                self.tm.start_training(
                    epochs=2,
                    input_size=64,
                    batch_size=8,
                    resume_from_latest=False,
                    use_augmentation=False,
                    on_complete=None,
                )
                # Wait until worker is guaranteed paused right before consuming its first run input
                self.assertTrue(worker_paused.wait(timeout=30.0), "Worker did not reach input read barrier")

                # Mutate and completely wipe the live dataset BEFORE the worker opens any image
                self.tm.update_sample_annotation(samples_before[0]["sample_id"], new_label="absent")
                self.tm.delete_sample(samples_before[1]["sample_id"])
                self.tm.clear_dataset()
                self.assertEqual(self.tm.get_dataset_summary()["total_samples"], 0)
                for p in original_paths:
                    assert p is not None
                    self.assertFalse(p.exists())
            finally:
                release_worker.set()

            status = _wait_for_training(self.tm)

        self.assertEqual(status["state"], "COMPLETED", f"Training failed after synchronized clear: {status}")
        latest = status["latest_run"]
        self.assertIsNotNone(latest)
        self.assertEqual(latest["train_samples"] + latest["val_samples"] + latest["test_samples"], 24)
        self.assertTrue(latest["onnx_parity_ok"])
        self.assertTrue(Path(latest["onnx_path"]).is_file())
        run_dir = Path(latest["onnx_path"]).parent
        self.assertFalse((run_dir / "_input_snapshot").exists())

    # ------------------------------------------------------------------
    # BE-27: Persist group provenance and keep trained groups out of warm-start holdouts
    # ------------------------------------------------------------------
    def test_be27_warm_start_provenance_enforces_disjoint_holdouts(self) -> None:
        import torch

        _populate_real_tubes(self.tm, ["TUBE_A", "TUBE_B", "TUBE_C", "TUBE_D"], samples_per_class=2)

        # Run 1: fresh training on 4 tubes -> default train=[TUBE_A, TUBE_B], val=[TUBE_C], test=[TUBE_D]
        self.tm.start_training(epochs=2, input_size=64, batch_size=8, resume_from_latest=False, use_augmentation=False)
        st1 = _wait_for_training(self.tm)
        self.assertEqual(st1["state"], "COMPLETED", st1)
        run1 = st1["latest_run"]
        self.assertEqual(run1["train_groups"], ["TUBE_A", "TUBE_B"])
        self.assertEqual(run1["seen_groups"], ["TUBE_A", "TUBE_B"])
        self.assertEqual(run1["val_groups"], ["TUBE_C"])
        self.assertEqual(run1["test_groups"], ["TUBE_D"])
        self.assertEqual(run1["evaluation_kind"], "independent_holdout")
        self.assertTrue(run1["independent_holdout_verified"])

        # Verify checkpoint itself stores provenance
        run1_dir = Path(run1["onnx_path"]).parent
        ckpt1 = torch.load(run1_dir / "label_classifier.pth", map_location="cpu", weights_only=True)
        self.assertEqual(ckpt1["train_groups"], ["TUBE_A", "TUBE_B"])
        self.assertEqual(ckpt1["seen_groups"], ["TUBE_A", "TUBE_B"])

        # Modify checkpoint provenance to pretend Run 1 trained on [TUBE_C, TUBE_D] (which default split would pick as holdouts!)
        cd_rows = [r for r in self.tm._read_manifest_rows() if r["physical_tube_id"] in {"TUBE_C", "TUBE_D"}]
        cd_sids = sorted(r["sample_id"] for r in cd_rows)
        cd_digests = sorted(self.tm._row_image_digest(r) for r in cd_rows)
        ckpt1["train_groups"] = ["TUBE_C", "TUBE_D"]
        ckpt1["seen_groups"] = ["TUBE_C", "TUBE_D"]
        ckpt1["train_sample_ids"] = cd_sids
        ckpt1["seen_sample_ids"] = cd_sids
        ckpt1["train_image_digests"] = cd_digests
        ckpt1["seen_image_digests"] = cd_digests
        torch.save(ckpt1, run1_dir / "label_classifier.pth")
        meta1 = json.loads((run1_dir / "metrics.json").read_text(encoding="utf-8"))
        meta1["train_groups"] = ["TUBE_C", "TUBE_D"]
        meta1["seen_groups"] = ["TUBE_C", "TUBE_D"]
        meta1["train_sample_ids"] = cd_sids
        meta1["seen_sample_ids"] = cd_sids
        meta1["train_image_digests"] = cd_digests
        meta1["seen_image_digests"] = cd_digests
        (run1_dir / "metrics.json").write_text(json.dumps(meta1), encoding="utf-8")

        # Run 2 (resume_from_latest=True): MUST NOT place TUBE_C or TUBE_D into val/test -> automatically picks TUBE_A/TUBE_B
        self.tm.start_training(epochs=2, input_size=64, batch_size=8, resume_from_latest=True, use_augmentation=False)
        st2 = _wait_for_training(self.tm)
        self.assertEqual(st2["state"], "COMPLETED", st2)
        run2 = st2["latest_run"]
        self.assertEqual(run2["resumed_from"], run1["model_version"])
        self.assertTrue(set(run2["val_groups"]).isdisjoint({"TUBE_C", "TUBE_D"}))
        self.assertTrue(set(run2["test_groups"]).isdisjoint({"TUBE_C", "TUBE_D"}))
        self.assertEqual(set(run2["val_groups"]) | set(run2["test_groups"]), {"TUBE_A", "TUBE_B"})

        # Now delete TUBE_B so only 1 unseen tube (TUBE_A) remains relative to {TUBE_C, TUBE_D}
        to_remove = [s["sample_id"] for s in self.tm.list_samples() if s["physical_tube_id"] == "TUBE_B"]
        self.tm.batch_delete_samples(sample_ids=to_remove)

        # Run 3 (resume_from_latest=True): only 1 unseen group left -> MUST reject incompatible warm-start cleanly!
        self.tm.start_training(epochs=2, input_size=64, batch_size=8, resume_from_latest=True, use_augmentation=False)
        st3 = _wait_for_training(self.tm)
        self.assertEqual(st3["state"], "FAILED")
        self.assertIn("增量微调", str(st3["error"]))

        # Run 4 (resume_from_latest=False): fresh training on the 3 remaining tubes succeeds cleanly
        self.tm.start_training(epochs=2, input_size=64, batch_size=8, resume_from_latest=False, use_augmentation=False)
        st4 = _wait_for_training(self.tm)
        self.assertEqual(st4["state"], "COMPLETED", st4)
        self.assertIsNone(st4["latest_run"]["resumed_from"])

        # Legacy checkpoint lacking train_groups/seen_groups provenance must be rejected on warm-start
        legacy_root = self.data_root / "legacy_ckpt_test"
        legacy_tm = self._make_manager(legacy_root)
        _populate_real_tubes(legacy_tm, ["TUBE_L1", "TUBE_L2", "TUBE_L3"], samples_per_class=2)
        legacy_run_dir = legacy_root / "runs" / "label_classifier" / "mnetv3_legacy_no_prov"
        legacy_run_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": ckpt1["state_dict"], "classes": CLASSES, "input_size": 64}, legacy_run_dir / "label_classifier.pth")
        legacy_tm.start_training(epochs=2, input_size=64, batch_size=8, resume_from_latest=True, use_augmentation=False)
        st_legacy = _wait_for_training(legacy_tm)
        self.assertEqual(st_legacy["state"], "FAILED")
        self.assertIn("缺少历史训练分组溯源", str(st_legacy["error"]))

    # ------------------------------------------------------------------
    # BE-28: Do not deploy an export that fails ONNX parity & isolate failed artifact
    # ------------------------------------------------------------------
    def test_be28_onnx_parity_failure_blocks_callback_and_isolates_artifact(self) -> None:
        import onnxruntime as ort

        _populate_real_tubes(self.tm, ["TUBE_P1", "TUBE_P2", "TUBE_P3"], samples_per_class=2)

        # First produce one verified run so we confirm a later parity-failed run cannot displace it
        self.tm.start_training(epochs=2, input_size=64, batch_size=8, resume_from_latest=False, use_augmentation=False)
        st_good = _wait_for_training(self.tm)
        self.assertEqual(st_good["state"], "COMPLETED", st_good)
        good_version = st_good["latest_run"]["model_version"]

        callback_invoked = []

        def spy_on_complete(result: dict) -> bool:
            callback_invoked.append(result)
            return True

        real_ort_run = ort.InferenceSession.run

        def perturbed_ort_run(sess, output_names, input_feed, *args, **kwargs):
            outs = real_ort_run(sess, output_names, input_feed, *args, **kwargs)
            return [outs[0] + 2.5]

        with mock.patch.object(ort.InferenceSession, "run", autospec=True, side_effect=perturbed_ort_run):
            self.tm.start_training(
                epochs=2,
                input_size=64,
                batch_size=8,
                resume_from_latest=False,
                use_augmentation=False,
                on_complete=spy_on_complete,
            )
            st_bad = _wait_for_training(self.tm)

        self.assertEqual(callback_invoked, [], "on_complete MUST NOT be called when ONNX parity fails")
        self.assertEqual(st_bad["state"], "FAILED")
        self.assertFalse(st_bad["exported"])
        self.assertFalse(st_bad["activated"])
        self.assertEqual(st_bad["activation_status"], "blocked_parity_failed")

        bad_run = st_bad["latest_run"]
        self.assertIsNotNone(bad_run)
        self.assertFalse(bad_run["onnx_parity_ok"])
        self.assertFalse(bad_run["onnx_parity_passed"])
        self.assertIsNone(bad_run["onnx_path"])
        self.assertIsNone(bad_run["onnx_sha256"])
        failed_onnx = Path(bad_run["failed_onnx_path"])
        self.assertTrue(failed_onnx.is_file())
        self.assertFalse((failed_onnx.parent / "label_classifier.onnx").exists())

        # Verified run selector must skip the failed run and select only the prior verified run
        verified_runs = self.tm.find_verified_onnx_runs()
        self.assertEqual([r["model_version"] for r in verified_runs], [good_version])

    # ------------------------------------------------------------------
    # BE-29: Report model activation only after on_complete succeeds
    # ------------------------------------------------------------------
    def test_be29_on_complete_ordering_and_activation_failure_reporting(self) -> None:
        _populate_real_tubes(self.tm, ["TUBE_A1", "TUBE_A2", "TUBE_A3"], samples_per_class=2)

        # Case 1: Successful activation — status must STILL be in progress while on_complete runs
        observed_during_cb: dict = {}

        def cb_ok(run_summary: dict) -> bool:
            observed_during_cb.update(self.tm.get_training_status())
            return True

        self.tm.start_training(
            epochs=2,
            input_size=64,
            batch_size=8,
            resume_from_latest=False,
            use_augmentation=False,
            on_complete=cb_ok,
        )
        st_ok = _wait_for_training(self.tm)
        self.assertEqual(observed_during_cb["state"], "EXPORTING_ONNX")
        self.assertFalse(observed_during_cb["activated"])
        self.assertEqual(st_ok["state"], "COMPLETED", st_ok)
        self.assertTrue(st_ok["exported"])
        self.assertTrue(st_ok["activated"])
        self.assertEqual(st_ok["activation_status"], "activated")
        self.assertTrue(st_ok["latest_run"]["activated"])

        # Case 2: on_complete returns False -> artifact exported, activation failed (NOT falsely COMPLETED/hot-loaded)
        self.tm.start_training(
            epochs=2,
            input_size=64,
            batch_size=8,
            resume_from_latest=False,
            use_augmentation=False,
            on_complete=lambda _: False,
        )
        st_false = _wait_for_training(self.tm)
        self.assertEqual(st_false["state"], "FAILED")
        self.assertTrue(st_false["exported"])
        self.assertFalse(st_false["activated"])
        self.assertEqual(st_false["activation_status"], "activation_failed")
        self.assertTrue(Path(st_false["latest_run"]["onnx_path"]).is_file())
        self.assertFalse(st_false["latest_run"]["activated"])
        self.assertNotIn("已热加载", st_false["message"])

        # Case 3: on_complete raises exception -> activation failure recorded clearly while preserving exported artifact
        def cb_raise(_: dict) -> bool:
            raise RuntimeError("ONNX session load error")

        self.tm.start_training(
            epochs=2,
            input_size=64,
            batch_size=8,
            resume_from_latest=False,
            use_augmentation=False,
            on_complete=cb_raise,
        )
        st_exc = _wait_for_training(self.tm)
        self.assertEqual(st_exc["state"], "FAILED")
        self.assertTrue(st_exc["exported"])
        self.assertFalse(st_exc["activated"])
        self.assertIn("ONNX session load error", str(st_exc["error"]))
        self.assertTrue(Path(st_exc["latest_run"]["onnx_path"]).is_file())

        # Case 4: on_complete=None -> COMPLETED as exported_only, never claiming hot-loaded
        self.tm.start_training(
            epochs=2,
            input_size=64,
            batch_size=8,
            resume_from_latest=False,
            use_augmentation=False,
            on_complete=None,
        )
        st_none = _wait_for_training(self.tm)
        self.assertEqual(st_none["state"], "COMPLETED")
        self.assertTrue(st_none["exported"])
        self.assertFalse(st_none["activated"])
        self.assertEqual(st_none["activation_status"], "exported_only")
        self.assertNotIn("已热加载", st_none["message"])

    # ------------------------------------------------------------------
    # R2-13: Preserve crop-space boxes when rebuilding COCO from manifest
    # ------------------------------------------------------------------
    def test_r2_13_rebuild_coco_preserves_subcrop_boxes_and_legacy_fallback(self) -> None:
        s_sub = self.tm.capture_sample(
            crop_bgr=_make_dummy_crop(601),
            label="crossed_out",
            physical_tube_id="TUBE_R13",
            box_xyxy=[0.30, 0.35, 0.50, 0.65],
            crop_bounds_xyxy=[0.20, 0.20, 0.60, 0.80],
        )
        s_legacy = self.tm.capture_sample(
            crop_bgr=_make_dummy_crop(602),
            label="complete",
            physical_tube_id="TUBE_R13",
            box_xyxy=[0.10, 0.15, 0.50, 0.55],
        )
        self.assertEqual(s_sub["crop_box_xyxy"], "0.2500,0.2500,0.7500,0.7500")

        # Strip crop_box_xyxy from s_legacy in manifest CSV to simulate a pre-R2-13 legacy row
        manifest_path = self.data_root / "dataset" / "samples_manifest.csv"
        with manifest_path.open("r", newline="", encoding="utf-8-sig") as rf:
            rows = list(csv.DictReader(rf))
        for row in rows:
            if row["sample_id"] == s_legacy["sample_id"]:
                row["crop_box_xyxy"] = ""
        self.tm._write_all_manifest_rows(rows)

        coco_path = self.data_root / "dataset" / "annotations_coco.json"
        coco_path.unlink()

        rebuilt = self.tm.rebuild_coco_from_manifest()
        self.assertEqual(rebuilt["rebuilt_annotations"], 2)
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
        img_by_sid = {im["sample_id"]: im for im in coco["images"]}
        ann_by_img_id = {an["image_id"]: an for an in coco["annotations"]}

        sub_ann = ann_by_img_id[img_by_sid[s_sub["sample_id"]]["id"]]
        self.assertEqual(sub_ann["bbox"], [56.0, 36.0, 112.0, 72.0])
        self.assertEqual(sub_ann["area"], 8064.0)

        legacy_ann = ann_by_img_id[img_by_sid[s_legacy["sample_id"]]["id"]]
        self.assertEqual(legacy_ann["bbox"], [0.0, 0.0, 224.0, 144.0])

    # ------------------------------------------------------------------
    # R2-14: Invalid UTF-8 or COCO write failure returns committed sample with coco_synced=False
    # ------------------------------------------------------------------
    def test_r2_14_invalid_utf8_and_coco_write_failure_preserve_committed_sample(self) -> None:
        coco_path = self.data_root / "dataset" / "annotations_coco.json"
        coco_path.parent.mkdir(parents=True, exist_ok=True)
        bad_utf8 = b"\xff\xfe\x80\x81not-utf8"
        coco_path.write_bytes(bad_utf8)

        # Must NOT raise UnicodeDecodeError/ValueError after committing sample; preserve invalid file untouched
        committed = self.tm.upload_sample_bytes(
            image_bytes=_encode_jpg(_make_dummy_crop(610)),
            label="complete",
            physical_tube_id="TUBE_R14",
        )
        self.assertFalse(committed["coco_synced"])
        self.assertEqual(coco_path.read_bytes(), bad_utf8)
        self.assertEqual(len(self.tm.list_samples()), 1)

        # Restore valid COCO via rebuild, then simulate OSError during atomic write
        self.tm.rebuild_coco_from_manifest()
        valid_before = coco_path.read_text(encoding="utf-8")
        with mock.patch.object(tm_mod, "_write_json_atomic", side_effect=OSError("COCO read-only")):
            committed2 = self.tm.upload_sample_bytes(
                image_bytes=_encode_jpg(_make_dummy_crop(611)),
                label="crossed_out",
                physical_tube_id="TUBE_R14",
            )
        self.assertFalse(committed2["coco_synced"])
        self.assertEqual(coco_path.read_text(encoding="utf-8"), valid_before)
        self.assertEqual(len(self.tm.list_samples()), 2)

        # Subsequent healthy write succeeds with coco_synced=True
        committed3 = self.tm.upload_sample_bytes(
            image_bytes=_encode_jpg(_make_dummy_crop(612)),
            label="covered",
            physical_tube_id="TUBE_R14",
        )
        self.assertTrue(committed3["coco_synced"])

    # ------------------------------------------------------------------
    # R2-15: Keep live-source lineage stable across repeated bootstrap calls & split
    # ------------------------------------------------------------------
    def test_r2_15_bootstrap_source_lineage_stable_across_calls_and_split(self) -> None:
        src_a = _make_dummy_crop(701, (240, 20, 240))
        src_b = _make_dummy_crop(702, (20, 240, 20))
        src_c = _make_dummy_crop(703, (240, 240, 20))
        src_d = _make_dummy_crop(704, (20, 120, 240))

        # Call 1 assigns A -> TUBE_G01; Call 2 passes [B, C, A, D] -> A MUST stay pinned to TUBE_G01!
        s1 = self.tm.bootstrap_standard_dataset(live_crops=[src_a])
        self.assertEqual(s1["live_crop_groups"], ["TUBE_G01"])
        s2 = self.tm.bootstrap_standard_dataset(live_crops=[src_b, src_c, src_a, src_d])
        self.assertEqual(set(s2["live_crop_groups"]), {"TUBE_G01", "TUBE_G02", "TUBE_G03", "TUBE_G04"})

        rows = self.tm._read_manifest_rows()
        groups_by_source: dict[str, set[str]] = {}
        for r in rows:
            dig = (r.get("source_digest") or "").strip()
            if dig:
                groups_by_source.setdefault(dig, set()).add(r["physical_tube_id"])
        self.assertEqual(len(groups_by_source), 4)
        for dig, grps in groups_by_source.items():
            self.assertEqual(len(grps), 1, f"Source {dig} leaked across multiple groups: {grps}")

        # Defense-in-depth in _select_group_split: if two groups share a source_digest, keep both in train!
        synthetic_rows = [
            {"sample_id": f"S_{g}_{c}", "label": c, "physical_tube_id": g, "source_digest": src}
            for g, src in [
                ("TUBE_G01", "src_shared"),
                ("TUBE_G02", "src_b"),
                ("TUBE_G03", "src_shared"),
                ("TUBE_G04", "src_d"),
            ]
            for c in CLASSES
        ]
        split = self.tm._select_group_split(synthetic_rows)
        self.assertEqual(set(split["train_groups"]), {"TUBE_G01", "TUBE_G03"})
        self.assertEqual(set(split["val_groups"]) | set(split["test_groups"]), {"TUBE_G02", "TUBE_G04"})

    # ------------------------------------------------------------------
    # R2-16 & R2-34: Include train-only ungrouped labels in coverage & O(1) per-pair scaling
    # ------------------------------------------------------------------
    def test_r2_16_and_r2_34_ungrouped_train_coverage_and_fast_large_group_split(self) -> None:
        # R2-16: TUBE_A has complete/crossed_out/covered; TUBE_B & TUBE_C have complete; UNGROUPED supplies absent
        for i in range(3):
            self.tm.capture_sample(_make_dummy_crop(710 + i), "complete", physical_tube_id="TUBE_A")
            self.tm.capture_sample(_make_dummy_crop(720 + i), "crossed_out", physical_tube_id="TUBE_A")
            self.tm.capture_sample(_make_dummy_crop(730 + i), "covered", physical_tube_id="TUBE_A")
            self.tm.capture_sample(_make_dummy_crop(740 + i), "absent", physical_tube_id=None)
        self.tm.capture_sample(_make_dummy_crop(750), "complete", physical_tube_id="TUBE_B")
        self.tm.capture_sample(_make_dummy_crop(751), "complete", physical_tube_id="TUBE_C")

        summary = self.tm.get_dataset_summary()
        self.assertEqual(summary["independent_tube_groups"], 3)
        self.assertEqual(summary["ungrouped_samples"], 3)
        self.assertTrue(summary["ready_to_train"], summary["readiness_reason"])

        split = self.tm._select_group_split(self.tm._read_manifest_rows())
        self.assertEqual(split["train_groups"], ["TUBE_A"])
        self.assertEqual(set(split["val_groups"]) | set(split["test_groups"]), {"TUBE_B", "TUBE_C"})
        self.assertEqual({r["label"] for r in split["train_rows"]}, set(CLASSES))

        # R2-34: 300 partial-class groups (no full-class holdouts) must resolve cleanly and quickly
        large_rows = []
        for g_idx in range(300):
            tk = f"TUBE_{g_idx:04d}"
            # Each group has 2 of the 4 classes so fallback pair search must run across many pairs
            c1 = CLASSES[g_idx % len(CLASSES)]
            c2 = CLASSES[(g_idx + 1) % len(CLASSES)]
            large_rows.append({"sample_id": f"L_{g_idx}_1", "label": c1, "physical_tube_id": tk, "source_digest": ""})
            large_rows.append({"sample_id": f"L_{g_idx}_2", "label": c2, "physical_tube_id": tk, "source_digest": ""})

        t0 = time.perf_counter()
        large_split = self.tm._select_group_split(large_rows)
        elapsed = time.perf_counter() - t0
        self.assertEqual({r["label"] for r in large_split["train_rows"]}, set(CLASSES))
        self.assertLess(elapsed, 1.5, f"300-group split took too long: {elapsed:.3f}s")

    # ------------------------------------------------------------------
    # R2-17: Exclude previously trained samples even after their tube ID changes
    # ------------------------------------------------------------------
    def test_r2_17_warm_start_excludes_reassigned_previously_trained_samples(self) -> None:
        _populate_real_tubes(self.tm, ["TUBE_W1", "TUBE_W2", "TUBE_W3"], samples_per_class=2)
        ungrouped_ids: list[str] = []
        for idx, cls in enumerate(CLASSES):
            s = self.tm.capture_sample(_make_dummy_crop(800 + idx), cls, physical_tube_id=None)
            ungrouped_ids.append(s["sample_id"])

        # Run 1 trains on TUBE_W1 + all 4 UNGROUPED samples (holdouts: TUBE_W2, TUBE_W3)
        self.tm.start_training(epochs=2, input_size=64, batch_size=8, resume_from_latest=False, use_augmentation=False)
        st1 = _wait_for_training(self.tm)
        self.assertEqual(st1["state"], "COMPLETED", st1)
        run1 = st1["latest_run"]
        self.assertEqual(run1["train_groups"], ["TUBE_W1"])
        for uid in ungrouped_ids:
            self.assertIn(uid, run1["seen_sample_ids"])

        # Engineer later reassigns those previously trained ungrouped samples into a brand-new tube ID
        for uid in ungrouped_ids:
            self.tm.update_sample_annotation(uid, physical_tube_id="TUBE_Z_REASSIGNED")

        # Run 2 (warm-start): TUBE_Z_REASSIGNED sorts last and has all 4 classes,
        # yet MUST NOT be placed into val or test because the checkpoint already trained on its samples!
        self.tm.start_training(epochs=2, input_size=64, batch_size=8, resume_from_latest=True, use_augmentation=False)
        st2 = _wait_for_training(self.tm)
        self.assertEqual(st2["state"], "COMPLETED", st2)
        run2 = st2["latest_run"]
        self.assertEqual(run2["resumed_from"], run1["model_version"])
        self.assertNotIn("TUBE_Z_REASSIGNED", run2["val_groups"])
        self.assertNotIn("TUBE_Z_REASSIGNED", run2["test_groups"])
        self.assertIn("TUBE_Z_REASSIGNED", run2["train_groups"])
        self.assertEqual(set(run2["val_groups"]) | set(run2["test_groups"]), {"TUBE_W2", "TUBE_W3"})

        # Delete TUBE_W3 so only 1 truly unseen tube (TUBE_W2) remains -> warm-start must reject cleanly!
        w3_ids = [s["sample_id"] for s in self.tm.list_samples() if s["physical_tube_id"] == "TUBE_W3"]
        self.tm.batch_delete_samples(sample_ids=w3_ids)
        self.tm.start_training(epochs=2, input_size=64, batch_size=8, resume_from_latest=True, use_augmentation=False)
        st3 = _wait_for_training(self.tm)
        self.assertEqual(st3["state"], "FAILED")
        self.assertIn("增量微调", str(st3["error"]))

    # ------------------------------------------------------------------
    # R2-18: Preserve successful activation when final metrics.json rewrite fails
    # ------------------------------------------------------------------
    def test_r2_18_activation_preserved_when_final_metrics_rewrite_fails(self) -> None:
        _populate_real_tubes(self.tm, ["TUBE_M1", "TUBE_M2", "TUBE_M3"], samples_per_class=2)
        real_write_json = tm_mod._write_json_atomic
        write_calls = {"count": 0}

        def flaky_metrics_write(path: Path, payload: dict) -> None:
            if path.name == "metrics.json":
                write_calls["count"] += 1
                # First write (pre-callback pending record) succeeds; subsequent post-activation rewrites fail
                if write_calls["count"] >= 2:
                    raise OSError("Simulated metrics disk lock")
            real_write_json(path, payload)

        with mock.patch.object(tm_mod, "_write_json_atomic", side_effect=flaky_metrics_write):
            self.tm.start_training(
                epochs=2,
                input_size=64,
                batch_size=8,
                resume_from_latest=False,
                use_augmentation=False,
                on_complete=lambda _: True,
            )
            st = _wait_for_training(self.tm)

        # Confirmed activation MUST remain COMPLETED & activated=True, NOT FAILED / activated=False!
        self.assertEqual(st["state"], "COMPLETED", st)
        self.assertTrue(st["exported"])
        self.assertTrue(st["activated"])
        self.assertEqual(st["activation_status"], "activated")
        self.assertFalse(st["metrics_persisted"])
        self.assertIn("Simulated metrics disk lock", str(st["metrics_write_error"]))
        # Automatic retry ran 3 times (1 initial pre-callback write + 3 post-activation retries = 4 calls)
        self.assertEqual(write_calls["count"], 4)

        # Once disk lock clears, explicit retry persists activated=True cleanly to metrics.json
        self.assertTrue(self.tm.retry_persist_latest_metrics())
        st_after = self.tm.get_training_status()
        self.assertTrue(st_after["metrics_persisted"])
        self.assertIsNone(st_after["metrics_write_error"])
        metrics_path = Path(st_after["latest_run"]["onnx_path"]).parent / "metrics.json"
        persisted_meta = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertTrue(persisted_meta["activated"])
        self.assertEqual(persisted_meta["activation_status"], "activated")


if __name__ == "__main__":
    unittest.main()
