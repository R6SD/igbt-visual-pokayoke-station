# Third-party notices

This is a project-level inventory for the local FastAPI UI, React frontend and ML tooling. The exact frontend dependency graph is pinned by `frontend/igbt-station-ui/package-lock.json`; the Python runtime is installed from `requirements-ui.txt`. This inventory is not a substitute for scanning the exact wheels and native libraries shipped in a release. Retain each package's bundled license/notice files in the release bundle.

| Component | License summary | Notes / source |
|---|---|---|
| PyTorch (`torch`) | Main project BSD-3-Clause; distributed packages include third-party components with their own notices | [PyTorch license and package metadata](https://github.com/pytorch/pytorch/blob/main/pyproject.toml) |
| torchvision | BSD-3-Clause | [torchvision repository](https://github.com/pytorch/vision) |
| ONNX | Apache-2.0 | [ONNX repository](https://github.com/onnx/onnx) |
| ONNX Runtime | MIT; wheels can bundle native third-party components | [ONNX Runtime license](https://github.com/microsoft/onnxruntime/blob/main/LICENSE) |
| Pillow | HPND / PIL license; verify exact installed distribution notices | [Pillow license](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| NumPy | BSD-3-Clause; binary wheels may bundle additional libraries | [NumPy license](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| Python | Python Software Foundation License; retain applicable notices | [Python license](https://docs.python.org/3/license.html) |
| FastAPI | MIT | [FastAPI license](https://github.com/fastapi/fastapi/blob/master/LICENSE) |
| Starlette | BSD-3-Clause | [Starlette repository](https://github.com/encode/starlette) |
| Uvicorn | BSD-3-Clause | [Uvicorn license](https://github.com/Kludex/uvicorn/blob/main/LICENSE.md) |
| Pydantic | MIT | [Pydantic license metadata](https://github.com/pydantic/pydantic/blob/main/pyproject.toml) |
| pydantic-core | MIT | Native validation/serialization dependency; [upstream license](https://github.com/pydantic/pydantic/blob/main/pydantic-core/LICENSE) |
| SQLite | Public domain dedication | [SQLite copyright](https://www.sqlite.org/copyright.html); Python `sqlite3` uses the platform/library build |
| OpenCV Python (camera probe) | Apache-2.0 for current OpenCV releases; package may bundle third-party codecs | [OpenCV license](https://github.com/opencv/opencv/blob/4.x/LICENSE) |
| React / React DOM | MIT | Runtime UI. [React license](https://github.com/facebook/react/blob/main/LICENSE) |
| shadcn/ui generated components | MIT for shadcn tooling; generated component source is copied into this project and may be modified | Source is in `frontend/igbt-station-ui/src/components/ui/`. [shadcn license](https://github.com/shadcn-ui/ui/blob/main/LICENSE.md) |
| Radix UI | MIT | Accessible primitives used by the generated Select, Badge and related controls. [Radix UI license](https://github.com/radix-ui/primitives/blob/main/LICENSE) |
| Tailwind CSS / @tailwindcss/vite | MIT | CSS generation and Vite integration. [Tailwind CSS license](https://github.com/tailwindlabs/tailwindcss/blob/main/LICENSE) |
| Vite / @vitejs/plugin-react | MIT | Frontend build tool and React integration. [Vite license](https://github.com/vitejs/vite/blob/main/LICENSE) |
| lucide-react | ISC | UI icon components. [Lucide license](https://github.com/lucide-icons/lucide/blob/main/LICENSE) |
| Geist Variable (`@fontsource-variable/geist`) | SIL Open Font License 1.1 | Locally bundled font files. [Fontsource package](https://github.com/fontsource/fontsource) |
| class-variance-authority | Apache-2.0 | shadcn component variant definitions. [License](https://github.com/joe-bell/cva/blob/main/LICENSE) |
| cn | MIT | Class name merging utility used by generated shadcn component source. [License](https://github.com/shadcn-ui/cn/blob/main/LICENSE) |
| tw-animate-css | MIT | Local utility animations used by shadcn component styles. [License](https://github.com/Wombosvideo/tw-animate-css/blob/main/LICENSE) |
| TypeScript | Apache-2.0 | Compile-time type checker; not loaded by the browser. [License](https://github.com/microsoft/TypeScript/blob/main/LICENSE.txt) |

No third-party pretrained weights are downloaded by `tools/train_classifier.py`. The MobileNetV3-Small architecture is instantiated with `weights=None`. The trained weights are derived from project data; dataset copyright, consent, retention and permitted-use rights must be documented separately. The frontend lockfile records resolved npm versions, including transitive packages. Before release, generate an SBOM and retain the resolved Python environment inventory and all wheel notices. The browser app loads only locally built assets from `/assets/`; it makes no CDN or external font requests at runtime.
