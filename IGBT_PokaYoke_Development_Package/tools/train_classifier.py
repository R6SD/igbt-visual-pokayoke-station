"""Grouped, from-scratch MobileNetV3-Small training for label ROI images.

Input CSV fields (all required):
sample_id,path,label,physical_tube_id,batch_id,video_group_id

label is complete/crossed_out/covered/absent/unknown_challenge. Challenge
samples are excluded from training and used only for exploratory rejection
threshold validation. No pretrained weights are downloaded or loaded.
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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CLASSES = ["complete", "crossed_out", "covered", "absent"]
CHALLENGE = "unknown_challenge"
LABELS = set(CLASSES + [CHALLENGE])
GROUP_FIELDS = ["physical_tube_id", "batch_id", "video_group_id"]
REQUIRED_FIELDS = ["sample_id", "path", "label", *GROUP_FIELDS]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = [field for field in REQUIRED_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")
        rows = []
        seen_ids: set[str] = set()
        root = csv_path.resolve().parent
        for line_no, original in enumerate(reader, start=2):
            row = {k: (v or "").strip() for k, v in original.items() if k is not None}
            for key in REQUIRED_FIELDS:
                if not row.get(key):
                    raise ValueError(f"Line {line_no}: {key} is required; group IDs may not be blank.")
            if row["label"] not in LABELS:
                raise ValueError(f"Line {line_no}: unsupported label {row['label']!r}; expected {sorted(LABELS)}")
            if row["sample_id"] in seen_ids:
                raise ValueError(f"Line {line_no}: duplicate sample_id {row['sample_id']!r}")
            seen_ids.add(row["sample_id"])
            image_path = Path(row["path"])
            if not image_path.is_absolute():
                image_path = (root / image_path).resolve()
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"Line {line_no}: image missing or unsupported: {image_path}")
            # Decode now so a corrupt image fails audit, before a long training run.
            try:
                from PIL import Image
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                raise ValueError(f"Line {line_no}: image cannot be decoded: {image_path}: {exc}") from exc
            row["path"] = str(image_path)
            row["sha256"] = sha256_file(image_path)
            rows.append(row)
    if not rows:
        raise ValueError("Input CSV contains no samples.")
    return rows


class UnionFind:
    def __init__(self, ids: list[str]):
        self.parent = {value: value for value in ids}
        self.rank = {value: 0 for value in ids}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def group_rows(rows: list[dict[str, str]], include_challenge: bool = True) -> dict[str, list[dict[str, str]]]:
    """Create connected components; any shared tube, batch, video, or hash stays together."""
    selected = [r for r in rows if include_challenge or r["label"] != CHALLENGE]
    ids = [r["sample_id"] for r in selected]
    uf = UnionFind(ids)
    owners: dict[tuple[str, str], str] = {}
    for row in selected:
        for field in GROUP_FIELDS:
            key = (field, row[field])
            if key in owners:
                uf.union(row["sample_id"], owners[key])
            else:
                owners[key] = row["sample_id"]
        key = ("sha256", row["sha256"])
        if key in owners:
            uf.union(row["sample_id"], owners[key])
        else:
            owners[key] = row["sample_id"]
    components: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        components[uf.find(row["sample_id"])].append(row)
    return dict(components)


def check_challenge_isolated(rows: list[dict[str, str]]) -> None:
    known = [r for r in rows if r["label"] != CHALLENGE]
    challenge = [r for r in rows if r["label"] == CHALLENGE]
    known_ids = {(field, r[field]) for r in known for field in GROUP_FIELDS}
    known_hashes = {r["sha256"] for r in known}
    for row in challenge:
        overlap = [field for field in GROUP_FIELDS if (field, row[field]) in known_ids]
        if overlap or row["sha256"] in known_hashes:
            raise ValueError(
                f"Challenge sample {row['sample_id']} overlaps known-data groups/hash ({overlap or 'same image'}). "
                "Keep rejection challenge material group-disjoint from train/val/test."
            )


def assign_splits(components: dict[str, list[dict[str, str]]], seed: int,
                  ratios: dict[str, float]) -> dict[str, str]:
    names = ["train", "val", "test"]
    if len(components) < 3:
        raise ValueError(
            f"Only {len(components)} independent group component(s); at least 3 are required for train/val/test. "
            "Do not split correlated frames to manufacture more groups."
        )
    rng = random.Random(seed)
    groups = list(components.items())
    rng.shuffle(groups)
    groups.sort(key=lambda item: len(item[1]), reverse=True)
    total = Counter(row["label"] for _, members in groups for row in members)
    targets = {part: {label: total[label] * ratios[part] for label in CLASSES} for part in names}
    counts = {part: Counter() for part in names}
    assignment: dict[str, str] = {}
    # Reserve one independent component per partition before balancing the remainder.
    for i, (component, members) in enumerate(groups[:3]):
        part = names[i]
        assignment[component] = part
        counts[part].update(row["label"] for row in members)

    def objective(candidate_part: str, members: list[dict[str, str]]) -> float:
        hypothetical = {p: counts[p].copy() for p in names}
        hypothetical[candidate_part].update(row["label"] for row in members)
        cost = 0.0
        for part in names:
            for label in CLASSES:
                target = max(targets[part][label], 1.0)
                cost += ((hypothetical[part][label] - targets[part][label]) / target) ** 2
        return cost

    for component, members in groups[3:]:
        part = min(names, key=lambda p: objective(p, members))
        assignment[component] = part
        counts[part].update(row["label"] for row in members)
    return assignment


def assigned_manifest(rows: list[dict[str, str]], seed: int, ratios: dict[str, float]):
    check_challenge_isolated(rows)
    known_components = group_rows([r for r in rows if r["label"] != CHALLENGE])
    split_by_component = assign_splits(known_components, seed, ratios)
    component_by_id = {}
    for component, members in known_components.items():
        for row in members:
            component_by_id[row["sample_id"]] = component

    canonical_by_hash: dict[str, dict[str, str]] = {}
    duplicate_of: dict[str, str] = {}
    for row in rows:
        previous = canonical_by_hash.get(row["sha256"])
        if previous is None:
            canonical_by_hash[row["sha256"]] = row
        elif previous["label"] != row["label"]:
            raise ValueError(
                f"Exact duplicate image has conflicting labels: {previous['sample_id']}={previous['label']} and "
                f"{row['sample_id']}={row['label']}"
            )
        else:
            duplicate_of[row["sample_id"]] = previous["sample_id"]

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        component = component_by_id.get(row["sample_id"], "challenge_" + row["sample_id"])
        is_duplicate = row["sample_id"] in duplicate_of
        split = "unknown_challenge" if row["label"] == CHALLENGE else split_by_component[component]
        output_rows.append({
            **row, "component_id": component, "split": split,
            "duplicate_of": duplicate_of.get(row["sample_id"], ""),
            "included_in_training": str(row["label"] in CLASSES and not is_duplicate).lower(),
            "data_role": "challenge_only" if row["label"] == CHALLENGE else ("duplicate_audit_only" if is_duplicate else "supervised"),
        })
    return output_rows, split_by_component


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def manifest_summary(manifest: list[dict[str, Any]], source_csv: Path) -> dict[str, Any]:
    by_split_label = {part: {label: sum(1 for r in manifest
                                        if r["split"] == part and r["label"] == label and not r["duplicate_of"])
                              for label in [*CLASSES, CHALLENGE]}
                      for part in ["train", "val", "test", "unknown_challenge"]}
    unique_hashes = len({r["sha256"] for r in manifest})
    return {
        "source_csv": str(source_csv.resolve()), "source_csv_sha256": sha256_file(source_csv),
        "rows_total": len(manifest), "unique_image_hashes": unique_hashes,
        "exact_duplicate_rows": sum(bool(r["duplicate_of"]) for r in manifest),
        "group_components_by_split": {
            part: len({r["component_id"] for r in manifest if r["split"] == part and r["label"] != CHALLENGE})
            for part in ["train", "val", "test"]
        },
        "class_counts_by_split": by_split_label,
        "group_fields": GROUP_FIELDS,
        "groups_shared_across_splits": False,
    }


def write_sha256s(run_dir: Path) -> None:
    sum_path = run_dir / "SHA256SUMS.txt"
    files = sorted(p for p in run_dir.rglob("*") if p.is_file() and p != sum_path)
    sum_path.write_text("".join(f"{sha256_file(p)}  {p.relative_to(run_dir).as_posix()}\n" for p in files), encoding="utf-8")


def image_tensor(paths: list[str], size: int):
    from PIL import Image
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
    images = []
    for filename in paths:
        with Image.open(filename) as source:
            images.append(transform(source.convert("RGB")))
    import torch
    return torch.stack(images)


def evaluate(model: Any, manifest: list[dict[str, Any]], split: str, size: int,
             batch_size: int, device: Any) -> dict[str, Any]:
    import torch
    label_to_idx = {label: i for i, label in enumerate(CLASSES)}
    rows = [r for r in manifest if r["split"] == split and r["label"] in CLASSES and not r["duplicate_of"]]
    if not rows:
        raise ValueError(f"No non-duplicate samples in split {split}.")
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[list[float]] = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            x = image_tensor([r["path"] for r in batch], size).to(device)
            probs = torch.softmax(model(x), dim=1).cpu()
            probabilities.extend(probs.tolist())
            y_true.extend(label_to_idx[r["label"]] for r in batch)
            y_pred.extend(int(i) for i in probs.argmax(dim=1).tolist())
    matrix = [[0 for _ in CLASSES] for _ in CLASSES]
    for truth, pred in zip(y_true, y_pred, strict=True):
        matrix[truth][pred] += 1
    per_class = {}
    for i, label in enumerate(CLASSES):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(matrix[row][i] for row in range(len(CLASSES)))
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
    complete_idx = label_to_idx["complete"]
    false_accept = {
        CLASSES[j]: matrix[complete_idx][j]
        for j in range(len(CLASSES)) if j != complete_idx
    }
    return {
        "split": split, "samples": len(rows),
        "accuracy": sum(matrix[i][i] for i in range(len(CLASSES))) / len(rows),
        "confusion_matrix": {"labels_order": CLASSES, "rows_true_cols_pred": matrix},
        "per_class": per_class,
        "complete_misaccepted_as_a_b_c": false_accept,
        "complete_misaccept_total": sum(false_accept.values()),
        "y_true": y_true, "y_pred": y_pred, "probabilities": probabilities,
        "sample_ids": [r["sample_id"] for r in rows],
    }


def rejection_validation(model: Any, manifest: list[dict[str, Any]], val: dict[str, Any],
                         size: int, batch_size: int, device: Any) -> dict[str, Any]:
    import torch
    val_probs = val["probabilities"]
    val_true = val["y_true"]
    val_pred = val["y_pred"]
    conf_by_class: dict[str, list[float]] = defaultdict(list)
    margin_by_class: dict[str, list[float]] = defaultdict(list)
    for truth, pred, probs in zip(val_true, val_pred, val_probs, strict=True):
        if truth == pred:
            ordered = sorted(probs, reverse=True)
            conf_by_class[CLASSES[truth]].append(ordered[0])
            margin_by_class[CLASSES[truth]].append(ordered[0] - ordered[1])
    thresholds = {}
    for label in CLASSES:
        cvals = sorted(conf_by_class[label])
        mvals = sorted(margin_by_class[label])
        if not cvals:
            thresholds[label] = {"min_probability": None, "min_margin": None, "basis": "no_correct_validation_samples"}
        else:
            idx = max(0, math.floor(0.05 * (len(cvals) - 1)))
            midx = max(0, math.floor(0.05 * (len(mvals) - 1)))
            thresholds[label] = {
                "min_probability": cvals[idx], "min_margin": mvals[midx],
                "basis": "5th percentile of correctly classified validation samples; exploratory only",
            }

    challenges = [r for r in manifest if r["label"] == CHALLENGE]
    scores = []
    if challenges:
        model.eval()
        with torch.inference_mode():
            for offset in range(0, len(challenges), batch_size):
                batch = challenges[offset:offset + batch_size]
                x = image_tensor([r["path"] for r in batch], size).to(device)
                scores.extend(torch.softmax(model(x), dim=1).cpu().tolist())
    rejected = 0
    for probs in scores:
        ordered = sorted(probs, reverse=True)
        pred = max(range(len(probs)), key=lambda i: probs[i])
        gate = thresholds[CLASSES[pred]]
        rejected += gate["min_probability"] is None or ordered[0] < gate["min_probability"] or ordered[0] - ordered[1] < gate["min_margin"]
    return {
        "status": "EXPLORATORY_NOT_FOR_PRODUCTION",
        "threshold_method": "per-class fifth percentile of correctly classified validation confidence and margin",
        "thresholds": thresholds,
        "unknown_challenge_count": len(challenges),
        "unknown_rejected": rejected,
        "unknown_rejection_rate": rejected / len(challenges) if challenges else None,
        "note": "Thresholds are diagnostics only; select and approve operating thresholds using independent challenge data and quality risk criteria.",
    }


def export_and_verify(model: Any, run_dir: Path, test_rows: list[dict[str, Any]], size: int,
                      device: Any, atol: float, rtol: float) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort
    import torch
    model.eval().cpu()
    onnx_path = run_dir / "label_classifier.onnx"
    example = image_tensor([r["path"] for r in test_rows[:min(16, len(test_rows))]], size)
    torch.onnx.export(
        model, example, str(onnx_path), input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(["logits"], {"image": example.numpy()})[0]
    with torch.inference_mode():
        expected = model(example).numpy()
    max_abs = float(np.max(np.abs(expected - actual)))
    max_rel = float(np.max(np.abs(expected - actual) / np.maximum(np.abs(expected), 1e-12)))
    passed = bool(np.allclose(expected, actual, atol=atol, rtol=rtol))
    result = {
        "status": "PASS" if passed else "FAIL", "samples_compared": len(test_rows[:min(16, len(test_rows))]),
        "atol": atol, "rtol": rtol, "max_absolute_difference": max_abs,
        "max_relative_difference": max_rel, "same_argmax_all": bool(np.array_equal(expected.argmax(1), actual.argmax(1))),
    }
    if not passed:
        raise RuntimeError(f"PyTorch/ONNX Runtime parity failed: {result}")
    return result


def run(args: argparse.Namespace) -> Path:
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import mobilenet_v3_small

    source_csv = Path(args.manifest).resolve()
    if args.input_size < 32:
        raise ValueError("input-size must be at least 32 pixels for MobileNetV3-Small.")
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, batch-size and learning-rate must be positive.")
    rows = read_rows(source_csv)
    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    if abs(sum(ratios.values()) - 1.0) > 1e-8 or any(v <= 0 for v in ratios.values()):
        raise ValueError("train/val/test ratios must be positive and sum to 1.")
    assigned, _ = assigned_manifest(rows, args.seed, ratios)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_seed{args.seed}"
    run_dir = Path(args.output_dir).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_csv(run_dir / "dataset_manifest.csv", assigned)
    (run_dir / "source_manifest.csv").write_bytes(source_csv.read_bytes())
    audit = manifest_summary(assigned, source_csv)
    (run_dir / "dataset_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.audit_only:
        write_sha256s(run_dir)
        print(f"Grouped data audit complete: {run_dir}")
        return run_dir

    if any(not any(r["label"] == label and r["split"] == "train" and not r["duplicate_of"] for r in assigned) for label in CLASSES):
        raise ValueError("Every one of the four classes must have a unique training image.")
    if not any(r["split"] == "val" and r["label"] in CLASSES and not r["duplicate_of"] for r in assigned):
        raise ValueError("No unique known-class validation samples after grouped split.")
    if not any(r["split"] == "test" and r["label"] in CLASSES and not r["duplicate_of"] for r in assigned):
        raise ValueError("No unique known-class test samples after grouped split.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
    model.to(device)

    class RoiDataset(Dataset):
        def __init__(self, selected: list[dict[str, Any]]):
            self.rows = selected
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, index):
            from PIL import Image
            from torchvision import transforms
            if not hasattr(self, "transform"):
                self.transform = transforms.Compose([
                    transforms.Resize((args.input_size, args.input_size), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(), transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
                ])
            row = self.rows[index]
            with Image.open(row["path"]) as im:
                x = self.transform(im.convert("RGB"))
            return x, CLASSES.index(row["label"])

    training_rows = [r for r in assigned if r["split"] == "train" and r["label"] in CLASSES and not r["duplicate_of"]]
    loader = DataLoader(RoiDataset(training_rows), batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    best_accuracy = -1.0
    best_state = None
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            total_seen += len(labels)
        val_result = evaluate(model, assigned, "val", args.input_size, args.batch_size, device)
        epoch_data = {"epoch": epoch, "train_loss": total_loss / max(total_seen, 1),
                      "val_accuracy": val_result["accuracy"]}
        history.append(epoch_data)
        print(f"epoch {epoch}/{args.epochs} train_loss={epoch_data['train_loss']:.5f} val_accuracy={val_result['accuracy']:.4f}")
        if val_result["accuracy"] > best_accuracy:
            best_accuracy = val_result["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No model checkpoint selected.")
    model.load_state_dict(best_state)
    model.to(device)
    torch.save({"state_dict": best_state, "classes": CLASSES, "input_size": args.input_size}, run_dir / "label_classifier.pth")
    val_result = evaluate(model, assigned, "val", args.input_size, args.batch_size, device)
    test_result = evaluate(model, assigned, "test", args.input_size, args.batch_size, device)
    rejection = rejection_validation(model, assigned, val_result, args.input_size, args.batch_size, device)
    test_rows = [r for r in assigned if r["split"] == "test" and r["label"] in CLASSES and not r["duplicate_of"]]
    parity = export_and_verify(model, run_dir, test_rows, args.input_size, device, args.onnx_atol, args.onnx_rtol)

    # JSON only contains serializable metrics; private prediction arrays stay out of summary.
    metrics = {
        "seed": args.seed, "device": str(device), "epochs": args.epochs, "best_val_accuracy": best_accuracy,
        "elapsed_seconds": round(time.time() - started, 2),
        "validation": {k: v for k, v in val_result.items() if k not in {"y_true", "y_pred", "probabilities", "sample_ids"}},
        "test": {k: v for k, v in test_result.items() if k not in {"y_true", "y_pred", "probabilities", "sample_ids"}},
        "unknown_challenge_rejection": rejection, "onnx_parity": parity,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "confusion_matrix_test.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true\\pred", *CLASSES])
        for label, values in zip(CLASSES, test_result["confusion_matrix"]["rows_true_cols_pred"], strict=True):
            writer.writerow([label, *values])
    (run_dir / "classification_report_test.json").write_text(
        json.dumps(test_result["per_class"], ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    metadata = {
        "model_name": "IGBT label ROI MobileNetV3-Small", "model_version": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "classes": CLASSES,
        "architecture": "torchvision.models.mobilenet_v3_small", "weights": "random initialization; weights=None",
        "input": {"width": args.input_size, "height": args.input_size, "color": "RGB",
                  "resize": "PIL/torchvision Resize((size,size), bilinear)", "crop": "none",
                  "normalization_mean": NORMALIZE_MEAN, "normalization_std": NORMALIZE_STD},
        "dataset_manifest_sha256": sha256_file(run_dir / "dataset_manifest.csv"),
        "source_manifest_sha256": sha256_file(run_dir / "source_manifest.csv"),
        "training": {"seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
                     "optimizer": "AdamW", "learning_rate": args.learning_rate},
        "torch_version": torch.__version__, "torchvision_version": __import__("torchvision").__version__,
        "onnxruntime_version": __import__("onnxruntime").__version__, "onnx_opset": 17,
        "reject_thresholds": rejection,
        "threshold_use": "Exploratory only; not validated for production or safety decisions.",
        "license_status": "Custom weights trained from project dataset; framework licenses listed separately. Dataset rights require project-owner confirmation.",
        "onnx_parity": parity,
    }
    metadata["weights_sha256"] = sha256_file(run_dir / "label_classifier.pth")
    metadata["onnx_sha256"] = sha256_file(run_dir / "label_classifier.onnx")
    (run_dir / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "MODEL_CARD.md").write_text(render_model_card(metadata, metrics, audit), encoding="utf-8")
    write_sha256s(run_dir)
    print(f"Training complete: {run_dir}")
    return run_dir


def render_model_card(metadata: dict[str, Any], metrics: dict[str, Any], audit: dict[str, Any]) -> str:
    test = metrics["test"]
    return f"""# Model card: {metadata['model_name']}

## Intended use

Research/prototype classification of a cropped IGBT label ROI into complete,
crossed_out, covered, or absent. This image classifier does not assess camera
validity, physical contact, operator process, or full-cycle acceptance. It must
not independently issue production PASS or drive an interlock.

## Model and preprocessing

- Architecture: {metadata['architecture']}
- Initialization: {metadata['weights']} (no pretrained asset downloaded)
- Input: {metadata['input']['width']}x{metadata['input']['height']} RGB; resize with bilinear interpolation, no crop
- Normalization mean/std: {metadata['input']['normalization_mean']} / {metadata['input']['normalization_std']}
- Classes: {', '.join(metadata['classes'])}
- Dataset manifest SHA256: `{metadata['dataset_manifest_sha256']}`
- Weight SHA256: `{metadata['weights_sha256']}`
- ONNX SHA256: `{metadata['onnx_sha256']}`

## Evaluation

- Test accuracy: {test['accuracy']:.4f} on {test['samples']} grouped-held-out samples
- `complete` misaccepted as A/B/C: {test['complete_misaccept_total']} total; per destination: `{test['complete_misaccepted_as_a_b_c']}`
- ONNX parity: {metadata['onnx_parity']['status']} (`atol={metadata['onnx_parity']['atol']}`, `rtol={metadata['onnx_parity']['rtol']}`)
- Unknown challenge rejection: {metrics['unknown_challenge_rejection'].get('unknown_rejected')}/{metrics['unknown_challenge_rejection'].get('unknown_challenge_count')} (exploratory only)

## Data and limitations

- Dataset split unit: connected groups of physical tube, production batch, video group, and exact image SHA256. No group should cross train/val/test.
- Exact duplicate rows are retained in the manifest but excluded from training.
- Unknown challenge images are held out from supervised training.
- Dataset rows: {audit['rows_total']}; unique hashes: {audit['unique_image_hashes']}.
- Random split and validation-derived rejection thresholds are not evidence of production reliability. Confirm group independence and evaluate complete cycles on a separately approved challenge set.
- Rights to captured imagery and annotations are not established by this model card.

## License and provenance

Weights were trained from project images using randomly initialized model parameters. Framework dependencies and their notices are recorded in the project third-party notice. The rights and permitted uses of the dataset must be documented by the data owner before external distribution.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate a grouped MobileNetV3-Small label ROI baseline.")
    parser.add_argument("manifest", help="CSV with sample_id,path,label,physical_tube_id,batch_id,video_group_id")
    parser.add_argument("--output-dir", default="runs/label_classifier")
    parser.add_argument("--audit-only", action="store_true", help="Validate data, detect duplicates, and emit grouped manifest without training.")
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--onnx-atol", type=float, default=1e-4)
    parser.add_argument("--onnx-rtol", type=float, default=1e-3)
    args = parser.parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
