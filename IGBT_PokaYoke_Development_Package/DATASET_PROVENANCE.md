# Dataset provenance register

The training utility does not create or claim ownership of source images. Each run copies the source manifest, hashes image bytes and emits a grouped `dataset_manifest.csv` with SHA256 and split assignment.

Before training or distribution, the data owner should complete the following for each dataset release:

| Field | Required record |
|---|---|
| Dataset release ID | `dataset_vNNN` and creation date |
| Source | Station/camera and capture procedure; confirm reference overlays are excluded |
| Rights | Owner, internal/external permitted use, consent/notice basis and approving role |
| Privacy | Operator anonymous IDs only; no unnecessary faces or direct identifiers |
| Retention | Raw image/video retention and deletion schedule |
| Labels | Annotator/reviewer, label definitions and adjudication record |
| Hand detector labels | Separate `hand_glove`/`marker`; for hand positives record white ESD glove type, anonymous reviewer and review date; train/val/test each contain reviewed station examples |
| Groups | Physical tube, production batch, video/capture group IDs |
| Integrity | Source manifest SHA256; per-image SHA256 in generated split manifest |
| Transformations | ROI configuration version and crop-generation software/version |

`unknown_challenge` images must remain isolated from all known-class tube, batch, video groups and image hashes. Their role is rejection validation; they are never included as a fifth supervised training class.

The local evidence store keeps uploaded original JPEG/PNG files under unique names, records SHA256, cycle ID, frame ID, capture timestamp, camera metadata, pipe color, operation event, anonymous operator ID and label state in SQLite. Evidence files are immutable through the UI/API (exclusive-create semantics); the application does not overwrite source files. This store is an operational evidence log and is not itself an approved training dataset until rights, retention, annotation and group-split fields above have been completed.
