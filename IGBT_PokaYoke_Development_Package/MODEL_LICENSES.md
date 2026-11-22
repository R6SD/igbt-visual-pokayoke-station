# Model and model-asset license register

| Asset | Source | License / rights | Current status |
|---|---|---|---|
| MobileNetV3-Small model implementation | `torchvision.models.mobilenet_v3_small` | torchvision project license; see `THIRD_PARTY_NOTICES.md` | Architecture code is a dependency, not a downloaded weight |
| Faster R-CNN MobileNetV3-Large FPN implementation | `torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn` | torchvision project license; see `THIRD_PARTY_NOTICES.md` | Detector is constructed with `weights=None`, `weights_backbone=None` |
| Initial weights | Constructed with `weights=None` | No pretrained model asset is used | Verified by training code |
| Project-trained classifier weights | `runs/label_classifier/<run-id>/label_classifier.pth` | Derived from project dataset | Rights inherit dataset rights; record manifest hash/model card per run |
| ONNX export | `runs/label_classifier/<run-id>/label_classifier.onnx` | Export of project-trained weights | Same data/weight rights; record SHA256 per run |
| Any later pretrained checkpoint | Not configured | Must assess exact URL, version, model card, training data and license before use | Not approved by default |

Framework source code licensing does not establish rights to training images or pretrained weights. Do not distribute trained weights until data-owner permission and the exact runtime dependency notices are recorded.

The local UI model registry records a model name/version, SHA256 and an explicit license review state. Registration does not assert that a model is cleared for deployment. Use `needs_review` or `unknown` until the precise architecture implementation, checkpoint origin, training-data rights and downstream distribution terms have been reviewed.
