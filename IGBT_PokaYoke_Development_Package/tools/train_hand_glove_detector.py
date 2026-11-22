"""Train a from-scratch torchvision hand_glove/marker detector from COCO data.

Requires COCO image metadata for capture_group/tube/batch IDs and a human
review attestation that positive hand_glove examples show real white station
gloves. Enforces such examples independently in train, val and test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CLASSES = ["background", "hand_glove", "marker"]
GROUP_FIELDS = ["physical_tube_id", "batch_id", "video_group_id"]
BASE_PROVENANCE_FIELDS = ["station_id", "operator_anon_id"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class UF:
    def __init__(self):
        self.p: dict[str, str] = {}
    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self, a: str, b: str) -> None:
        x, y = self.find(a), self.find(b)
        if x != y:
            self.p[y] = x


def load_coco(path: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    from PIL import Image
    data = json.loads(path.read_text(encoding="utf-8"))
    categories = {int(c["id"]): c["name"] for c in data.get("categories", [])}
    if "hand_or_pen" in categories.values():
        raise ValueError("Merged class 'hand_or_pen' is prohibited; annotate hand_glove and marker separately.")
    if categories.get(1) != "hand_glove" or categories.get(2) != "marker":
        raise ValueError("COCO category IDs must be 1=hand_glove and 2=marker.")
    annotations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ann in data.get("annotations", []):
        image_id = str(ann["image_id"])
        category = int(ann["category_id"])
        if category not in (1, 2):
            raise ValueError(f"Annotation category_id {category} is not hand_glove or marker.")
        x, y, w, h = [float(v) for v in ann["bbox"]]
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid bbox on annotation {ann.get('id')}")
        annotations[image_id].append({"category_id": category, "bbox": [x, y, w, h]})
    rows = []
    images = data.get("images", [])
    if not images:
        raise ValueError("COCO file has no images.")
    root = path.resolve().parent
    seen = set()
    for image in images:
        image_id = str(image["id"])
        if image_id in seen:
            raise ValueError(f"Duplicate COCO image id {image_id}")
        seen.add(image_id)
        for field in [*GROUP_FIELDS, *BASE_PROVENANCE_FIELDS]:
            if not str(image.get(field, "")).strip():
                raise ValueError(f"Image {image_id}: missing required provenance field {field}.")
        if not isinstance(image.get("worksite_white_glove"), bool):
            raise ValueError(f"Image {image_id}: worksite_white_glove must be an explicit true/false review field.")
        filename = Path(image["file_name"])
        if not filename.is_absolute():
            filename = (root / filename).resolve()
        if not filename.is_file():
            raise ValueError(f"Image {image_id}: file not found: {filename}")
        with Image.open(filename) as im:
            im.verify()
        with Image.open(filename) as im:
            actual_w, actual_h = im.size
        if int(image.get("width", actual_w)) != actual_w or int(image.get("height", actual_h)) != actual_h:
            raise ValueError(f"Image {image_id}: declared COCO width/height do not match the image file.")
        for ann in annotations.get(image_id, []):
            x, y, bw, bh = ann["bbox"]
            if not all(math.isfinite(v) for v in (x, y, bw, bh)) or x < 0 or y < 0 or x + bw > actual_w or y + bh > actual_h:
                raise ValueError(f"Image {image_id}: bbox {ann['bbox']} is non-finite or outside image bounds {actual_w}x{actual_h}.")
        has_hand_box = any(ann["category_id"] == 1 for ann in annotations.get(image_id, []))
        if has_hand_box:
            if image.get("worksite_white_glove") is not True or "white" not in str(image.get("glove_type", "")).lower():
                raise ValueError(f"Image {image_id}: hand_glove positive requires a reviewed actual white station glove.")
            for field in ["white_glove_reviewed_by", "white_glove_reviewed_at"]:
                if not str(image.get(field, "")).strip():
                    raise ValueError(f"Image {image_id}: hand_glove positive requires {field} reviewer provenance.")
        row = {**image, "id": image_id, "file_name": str(filename), "sha256": sha256_file(filename)}
        rows.append(row)
    unknown_images = set(annotations) - seen
    if unknown_images:
        raise ValueError(f"Annotations refer to unknown image IDs: {sorted(unknown_images)[:10]}")
    return rows, annotations, data["categories"]


def split_images(rows: list[dict[str, Any]], seed: int, ratios: tuple[float, float, float]):
    if abs(sum(ratios) - 1.0) > 1e-8 or any(r <= 0 for r in ratios):
        raise ValueError("Split ratios must be positive and sum to 1.")
    uf = UF()
    owners: dict[tuple[str, str], str] = {}
    for row in rows:
        image_id = row["id"]
        for field in GROUP_FIELDS:
            key = (field, str(row[field]))
            if key in owners:
                uf.union(image_id, owners[key])
            else:
                owners[key] = image_id
        key = ("sha256", row["sha256"])
        if key in owners:
            uf.union(image_id, owners[key])
        else:
            owners[key] = image_id
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[uf.find(row["id"])].append(row)
    if len(groups) < 3:
        raise ValueError("Fewer than 3 independent tube/batch/video/hash groups; cannot make train/val/test safely.")
    items = list(groups.items())
    random.Random(seed).shuffle(items)
    items.sort(key=lambda pair: len(pair[1]), reverse=True)
    parts = ["train", "val", "test"]
    assignment = {}
    sizes = [0, 0, 0]
    # Seed each split with an independent component, then balance remaining image counts.
    for index, (group, members) in enumerate(items[:3]):
        assignment[group] = parts[index]
        sizes[index] += len(members)
    total = len(rows)
    targets = [total * r for r in ratios]
    for group, members in items[3:]:
        index = min(range(3), key=lambda i: ((sizes[i] + len(members) - targets[i]) / max(targets[i], 1)) ** 2)
        assignment[group] = parts[index]
        sizes[index] += len(members)
    for row in rows:
        row["group_component_id"] = uf.find(row["id"])
        row["split"] = assignment[row["group_component_id"]]
    return rows


class CocoDataset:
    def __init__(self, rows, annotations):
        self.rows, self.annotations = rows, annotations
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        import torch
        from PIL import Image
        from torchvision.transforms.functional import pil_to_tensor
        row = self.rows[index]
        with Image.open(row["file_name"]) as im:
            image = pil_to_tensor(im.convert("RGB")).float().div_(255.0)
        h, w = image.shape[-2:]
        boxes, labels = [], []
        for ann in self.annotations.get(row["id"], []):
            x, y, bw, bh = ann["bbox"]
            x1, y1, x2, y2 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(ann["category_id"])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch, strict=True))


def box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    ba = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(aa + ba - inter, 1e-12)


def evaluate_detector(model, dataset, rows, annotations, device, score_threshold: float = 0.5):
    import torch
    counts = {1: {"tp": 0, "fp": 0, "fn": 0}, 2: {"tp": 0, "fp": 0, "fn": 0}}
    model.eval()
    with torch.inference_mode():
        for i in range(len(dataset)):
            image, target = dataset[i]
            pred = model([image.to(device)])[0]
            for category in (1, 2):
                gt_boxes = target["boxes"][target["labels"] == category].tolist()
                preds = [
                    (float(score), box.tolist())
                    for score, label_id, box in zip(pred["scores"], pred["labels"], pred["boxes"], strict=True)
                    if int(label_id) == category and float(score) >= score_threshold
                ]
                preds.sort(reverse=True, key=lambda item: item[0])
                used = set()
                for _, pbox in preds:
                    candidates = [(box_iou(pbox, gt), j) for j, gt in enumerate(gt_boxes) if j not in used]
                    best_iou, best_j = max(candidates, default=(0.0, -1))
                    if best_iou >= 0.5:
                        counts[category]["tp"] += 1
                        used.add(best_j)
                    else:
                        counts[category]["fp"] += 1
                counts[category]["fn"] += len(gt_boxes) - len(used)
    report = {}
    for category, name in [(1, "hand_glove"), (2, "marker")]:
        c = counts[category]
        precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[name] = {**c, "precision_iou50": precision, "recall_iou50": recall, "f1_iou50": f1}
    return report


def run(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    coco_path = Path(args.coco).resolve()
    rows, annotations, categories = load_coco(coco_path)
    rows = split_images(rows, args.seed, (args.train_ratio, args.val_ratio, args.test_ratio))
    # Deduplicate byte-identical images after grouping; preserve every source row in the manifest.
    seen_hash = set()
    for row in rows:
        row["duplicate_of"] = ""
        if row["sha256"] in seen_hash:
            row["duplicate_of"] = "same_sha256"
        else:
            seen_hash.add(row["sha256"])
        row["has_site_white_glove"] = bool(row["worksite_white_glove"] is True and any(
            a["category_id"] == 1 for a in annotations.get(row["id"], [])))
    for split in ("train", "val", "test"):
        if not any(r["split"] == split and r["has_site_white_glove"] and not r["duplicate_of"] for r in rows):
            raise ValueError(f"{split} must include independently grouped, human-reviewed real white-glove examples from this station.")
    canonical_rows = [r for r in rows if not r["duplicate_of"]]
    # Identical bytes with inconsistent annotations are ambiguous and must not be silently accepted.
    hash_annotations = {}
    for row in canonical_rows:
        signature = sorted((a["category_id"], tuple(a["bbox"])) for a in annotations.get(row["id"], []))
        hash_annotations[row["sha256"]] = signature
    for row in rows:
        if row["duplicate_of"]:
            original = next(r for r in canonical_rows if r["sha256"] == row["sha256"])
            signature = sorted((a["category_id"], tuple(a["bbox"])) for a in annotations.get(row["id"], []))
            if hash_annotations[row["sha256"]] != signature:
                raise ValueError(f"Duplicate image {row['id']} has conflicting annotations.")
            row["duplicate_of"] = original["id"]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_seed{args.seed}"
    out = Path(args.output_dir).resolve() / run_id
    out.mkdir(parents=True, exist_ok=False)
    with (out / "dataset_manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = list(dict.fromkeys(k for row in rows for k in row if k != "license"))
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    counts = {part: {"images": sum(r["split"] == part and not r["duplicate_of"] for r in rows),
                     "site_white_glove_images": sum(r["split"] == part and r["has_site_white_glove"] and not r["duplicate_of"] for r in rows)}
              for part in ("train", "val", "test")}
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    train_rows = [r for r in canonical_rows if r["split"] == "train"]
    val_rows = [r for r in canonical_rows if r["split"] == "val"]
    test_rows = [r for r in canonical_rows if r["split"] == "test"]
    train_ds, val_ds, test_ds = [CocoDataset(rs, annotations) for rs in (train_rows, val_rows, test_rows)]
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None, num_classes=3).to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate,
                                momentum=0.9, weight_decay=0.0005)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    best_f1 = -1.0
    best_state = None
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_report = evaluate_detector(model, val_ds, val_rows, annotations, device)
        hand_f1 = val_report["hand_glove"]["f1_iou50"]
        history.append({"epoch": epoch, "train_loss_mean": float(np.mean(losses)), "val_hand_glove_f1_iou50": hand_f1})
        print(f"epoch {epoch}/{args.epochs} loss={history[-1]['train_loss_mean']:.5f} val_hand_glove_f1_iou50={hand_f1:.4f}")
        if hand_f1 > best_f1:
            best_f1 = hand_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No detector checkpoint selected.")
    model.load_state_dict(best_state)
    model.to(device)
    checkpoint = out / "hand_glove_detector.pth"
    torch.save({"state_dict": best_state, "class_names": CLASSES}, checkpoint)
    test_report = evaluate_detector(model, test_ds, test_rows, annotations, device)
    metrics = {
        "test": test_report, "split_counts": counts, "history": history,
        "source_coco_sha256": sha256_file(coco_path),
        "marker_training_instances": sum(a["category_id"] == 2 for row in train_rows for a in annotations.get(row["id"], [])),
        "elapsed_seconds": round(time.time() - started, 2),
        "note": "Test includes actual site white ESD glove images as attested in COCO metadata. 2D detections are proximity/occlusion proxies, never physical-contact proof.",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "MODEL_CARD.md").write_text(
        f"# Hand glove detector model card\n\n- Architecture: torchvision Faster R-CNN MobileNetV3-Large FPN\n- Initialization: random; `weights=None`, `weights_backbone=None`\n- Classes: `hand_glove`, `marker` (separate IDs; no merged class)\n- Data: grouped by physical tube, production batch, video group, and SHA256\n- Source COCO SHA256: `{sha256_file(coco_path)}`\n- Real station white glove: required in train/val/test; recorded counts `{counts}`\n- Test metrics at IoU 0.5: `{test_report}`\n- Weight SHA256: `{sha256_file(checkpoint)}`\n- Intended use: glove box observations; candidate overlap is not proof of contact\n- No PASS/production output is implemented by this detector.\n",
        encoding="utf-8")
    hashes = []
    for p in sorted(out.iterdir()):
        if p.is_file():
            hashes.append(f"{sha256_file(p)}  {p.name}")
    (out / "SHA256SUMS.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    print(f"Detector training complete: {out}")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train separate hand_glove and marker detector with site-glove group checks.")
    parser.add_argument("coco", help="COCO JSON with provenance fields and bbox annotations")
    parser.add_argument("--output-dir", default="runs/hand_glove_detector")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size and learning-rate must be positive")
    try:
        run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
