# Model card template

每次真实训练运行会自动生成 `MODEL_CARD.md`，填写实际权重、数据清单、分组统计、指标、拒识挑战结果和 PyTorch/ONNX Runtime 一致性。未提供现场训练数据前，此模板不代表已有模型或性能。

## Expected model scope

- Architecture: MobileNetV3-Small
- Initialization: random (`weights=None`); no pretrained asset by default
- Input: label ROI image
- Classes: `complete`, `crossed_out`, `covered`, `absent`
- `unknown_challenge`: excluded from training; used only for exploratory rejection validation
- Use: research/prototype label ROI classification only
- Not a full-cycle decision system; does not establish `view_valid`, contact, or production PASS

Record dataset manifest SHA256, weight SHA256, ONNX SHA256, class order, preprocessing, framework versions/licenses, grouped split method, per-class test metrics, `complete` misaccept count, unknown rejection results and data rights before distributing a trained model.
