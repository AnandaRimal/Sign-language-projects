This project was completed with my team: Karan Bista and Nabin Kalel. Special thanks to my brother, Adarsha Rimal, for guiding and providing a good dataset.

# Sign Language Detection (GestureSpeak)

A full‑stack sign language and gesture detection app. The backend (Flask + TensorFlow + YOLO) exposes APIs for image classification, live webcam gesture streaming, object detection, and background removal. The frontend (React + Vite) provides an interactive UI and demos.

## Highlights

- Image classification with two models:
  - Custom CNN (`models/sign_language_cnn_model.h5`)
  - EfficientNet‑based model (`models/model_keras.h5`)
- Real‑time gesture detection using YOLOv8 (`models/best.pt`)
- Background removal via k‑means clustering
- Text‑to‑speech for detected gestures
- Ready‑to‑use React frontend wired to the backend APIs

## Folder structure

- `Backend/` — Flask app and REST endpoints (`backend.py`)
- `Frontend/` — React + Vite app (see `package.json` for scripts)
- `models/` — Pretrained model files: `best.pt`, `model_keras.h5`, `sign_language_cnn_model.h5`
- `accuracy_graphs/`, `model_training/`, `yolo_model_training/` — Notebooks and results

## Prerequisites

- Windows 10/11
- Python 3.10 (recommended) and PowerShell
- Node.js 18+ and npm
- Optional GPU acceleration
  - PyTorch CUDA 12.4 (matches the pinned versions)
  - TensorFlow‑GPU 2.10.0 uses older CUDA; if this is difficult to match on your machine, use CPU alternatives (notes below)

## Backend setup (Flask)

From the project root:

```powershell
# 1) Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

Notes on common installs:

- If PyTorch fails to install from `requirements.txt`, install explicitly using the official index for your platform:

  ```powershell
  # CUDA 12.4 build
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  
  # or CPU‑only build
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```

- If TensorFlow‑GPU 2.10.0 is hard to set up, you can switch to CPU TensorFlow:

  ```powershell
  pip uninstall -y tensorflow-gpu
  pip install tensorflow==2.10.0  # CPU build for TF 2.10 on Windows
  ```

- The YOLO model is initialized with `.cuda()` in `Backend/backend.py`. On CPU‑only machines, open that file and replace:

  ```python
  yolo_model = YOLO("../models/best.pt").cuda()
  ```
  with
  ```python
  yolo_model = YOLO("../models/best.pt")  # uses CPU by default
  ```

Run the backend:

```powershell
python .\Backend\backend.py
```

By default, the server starts on `http://127.0.0.1:5000`. Quick health check:

```powershell
curl http://127.0.0.1:5000/test
```

## Frontend setup (React + Vite)

The frontend fetches the backend at `http://127.0.0.1:5000` (hard‑coded in components), so start the backend first.

```powershell
cd .\Frontend
npm install
npm run dev
```

Open the URL shown by Vite (typically `http://127.0.0.1:5173`).

## Key endpoints

- `POST /classify-image`
  - Form‑data: `image` (file), `model_type` = `cnn` | `efficientnet`
  - Response: `{ model, class, confidence }`
- `GET /detect-gesture`
  - Streams annotated webcam frames (MJPEG)
- `POST /remove-background`
  - Form‑data: `image` (file)
  - Response: processed image (JPEG)
- `POST /object-detection`
  - Form‑data: `image` (file)
  - Response: `{ annotated_image (base64), classes: string[] }`
- `GET /test`
  - Health/status of loaded models

## Models

Place the following files under `models/`:

- `best.pt` — YOLOv8 weights
- `model_keras.h5` — EfficientNet‑based classifier
- `sign_language_cnn_model.h5` — Custom CNN classifier

These files are already present in this repository. If you replace them, keep the same filenames or update `Backend/backend.py` accordingly.

## Troubleshooting

- Webcam not working in Live Demo
  - Ensure a camera is connected and available; change `cv2.VideoCapture(0)` index if needed.
- GPU errors (CUDA/cuDNN)
  - Use the CPU instructions above or match CUDA versions for PyTorch/TensorFlow.
- CORS
  - CORS is enabled in the Flask app. Use `http://127.0.0.1:5000` as in the frontend code.
- Port conflicts
  - Change the Flask port in `backend.py` or Vite port via `--port`.

## License

This project is licensed under the terms in `LICENSE.txt`.

## Credits

- Team: Karan Bista, Nabin Kalel
- Special thanks: Adarsha Rimal — guidance and dataset support
- Owner/Maintainer: Ananda Rimal
