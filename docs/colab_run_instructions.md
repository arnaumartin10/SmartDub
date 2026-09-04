# MuseTalk en Colab

## Cel·la 1: muntar Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Cel·la 2: clonar el repositori amb el submòdul

```python
%cd /content
REPO_URL = "https://github.com/arnaumartin10/SmartDub.git"
!git clone --recurse-submodules "$REPO_URL" lipsync-pipeline
%cd /content/lipsync-pipeline
!git submodule update --init --recursive
```

## Avís obligatori

**Si qualsevol cel·la d’instal·lació o descàrrega mostra un ERROR, ATURA’T aquí i no continuïs amb les cel·les següents — copia l’error complet abans de seguir.**

## Cel·la 3: comprovar GPU i FFmpeg

```python
!nvidia-smi
!ffmpeg -version | head -n 2
import torch
print(f"Torch preinstal·lat: {torch.__version__}; CUDA: {torch.version.cuda}; disponible: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "CUDA no disponible: atura't i copia l'error complet."
```

## Cel·la 4: instal·lar dependències sense substituir Torch de Colab

```python
%cd /content/lipsync-pipeline
# Colab ja porta torch/torchvision/torchaudio compilats per la seva CUDA.
# No instal·lar requirements.txt complet: podria substituir Torch i CUDA.
import subprocess
import sys

def install(*packages, no_deps=False):
  command = [sys.executable, "-m", "pip", "install", "-q"]
  if no_deps:
    command.append("--no-deps")
  subprocess.check_call(command + list(packages))

# --no-deps evita que aquests paquets intentin reemplaçar Torch/CUDA de Colab.
install(
  "diffusers==0.30.2", "accelerate==0.28.0",
  "transformers==4.48.3", "huggingface_hub==0.36.2",
  "whisperx==3.8.6", "faster-whisper==1.2.0", "ctranslate2==4.8.2",
  "g2p_en==2.1.0", "mediapipe==0.10.35", "scenedetect==0.6.7", "soundfile==0.12.1",
  "librosa==0.10.2.post1", "einops==0.8.1", "omegaconf", "ffmpeg-python",
  no_deps=True,
)

# Reemplaça les wheels OpenCV antigues que poden haver vingut preinstal·lades a Colab.
# opencv-contrib-python també proporciona el mòdul cv2 i és el proveïdor que
# MediaPipe espera; no instal·lem opencv-python alhora.
subprocess.run(
  [sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-contrib-python"],
  check=False,
  stdout=subprocess.DEVNULL,
)
install("opencv-contrib-python==5.0.0.93", no_deps=True)

# ONNX Runtime és necessari per al VAD de faster-whisper. Aquest paquet no
# instal·la Torch, per tant es pot resoldre normalment.
install(
  "onnxruntime==1.20.1", "pyannote-audio==4.0.0", "torchcodec==0.7.0",
  "av", "pandas", "scipy", "tqdm", "pyyaml",
  "sentencepiece", "inflect==7.5.0", "distance==0.1.3",
)

# Verifica que l'stack s'ha instal·lat i que pip no ha substituït Torch.
import importlib.metadata as metadata
import torch
import numpy as np
import cv2
import mediapipe
import scenedetect
import onnxruntime
import faster_whisper
import whisperx
print(f"Torch: {torch.__version__}; CUDA: {torch.version.cuda}; disponible: {torch.cuda.is_available()}")
print("NumPy:", np.__version__)
print("OpenCV:", cv2.__version__)
print("MediaPipe:", mediapipe.__version__)
print("PySceneDetect:", scenedetect.__version__)
print("WhisperX:", metadata.version("whisperx"))
print("CTranslate2:", metadata.version("ctranslate2"))
print("ONNX Runtime:", metadata.version("onnxruntime"))
assert torch.cuda.is_available()
assert metadata.version("whisperx") == "3.8.6"
assert metadata.version("ctranslate2") == "4.8.2"
assert metadata.version("onnxruntime") == "1.20.1"
assert tuple(int(part) for part in cv2.__version__.split(".")[:2]) >= (5, 0)
assert tuple(int(part) for part in mediapipe.__version__.split(".")[:2]) >= (0, 10)
print("Imports: whisperx, faster_whisper, onnxruntime OK")
print("Imports: cv2, mediapipe, scenedetect OK")
print("Dependency installation completed successfully.")
```

## Cel·la 5: descarregar checkpoints MuseTalk

```python
%cd /content/lipsync-pipeline
!pip install -q --no-deps "huggingface_hub==0.36.2" gdown
!mkdir -p models/musetalk models/musetalkV15 models/sd-vae models/whisper

!hf download TMElyralab/MuseTalk \
  --local-dir models \
  --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth"
!hf download stabilityai/sd-vae-ft-mse \
  --local-dir models/sd-vae \
  --include "config.json" "diffusion_pytorch_model.bin"
!hf download openai/whisper-tiny \
  --local-dir models/whisper \
  --include "config.json" "pytorch_model.bin" "preprocessor_config.json"

from pathlib import Path
required_checkpoints = [
    Path("models/musetalkV15/musetalk.json"),
    Path("models/musetalkV15/unet.pth"),
    Path("models/sd-vae/config.json"),
    Path("models/sd-vae/diffusion_pytorch_model.bin"),
    Path("models/whisper/config.json"),
    Path("models/whisper/pytorch_model.bin"),
    Path("models/whisper/preprocessor_config.json"),
]
missing = [str(path) for path in required_checkpoints if not path.is_file()]
if missing:
    raise RuntimeError(f"Checkpoint download failed; missing files: {missing}")
print("Download completed. Checkpoint files:")
for path in required_checkpoints:
    print(f"  {path}: {path.stat().st_size} bytes")
```

## Cel·la 6: copiar els fitxers reals des de Drive

```python
%cd /content/lipsync-pipeline
from pathlib import Path

DRIVE_INPUT = Path('/content/drive/MyDrive/SmartDub/data/inputs')
Path('data/inputs').mkdir(parents=True, exist_ok=True)
!cp "$DRIVE_INPUT/sample.mp4" data/inputs/sample.mp4
!cp "$DRIVE_INPUT/sample_dub.wav" data/inputs/sample_dub.wav
!cp "$DRIVE_INPUT/sample_dub.txt" data/inputs/sample_dub.txt
!ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of default=noprint_wrappers=1 data/inputs/sample.mp4
!ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate,channels,duration -of default=noprint_wrappers=1 data/inputs/sample_dub.wav
```

## Cel·la 7: validar el wrapper i les entrades, sense carregar models

```python
%cd /content/lipsync-pipeline
from pathlib import Path
from src.generation.coarse_lipsync import CoarseLipSyncGenerator

assert Path('third_party/MuseTalk').exists()
assert Path('models/musetalkV15/unet.pth').exists()
assert Path('models/sd-vae/config.json').exists()
assert Path('models/whisper/config.json').exists()
print('Repository, checkpoints, and wrapper paths are ready.')
```

## Cel·la 8: executar preprocessing, alignment i MuseTalk

```python
%cd /content/lipsync-pipeline
!python scripts/generate_demo.py \
  --video data/inputs/sample.mp4 \
  --audio data/inputs/sample_dub.wav \
  --transcript data/inputs/sample_dub.txt \
  --checkpoint-dir models \
  --output data/outputs/musetalk_coarse.mp4
```

## Cel·la 9: inspeccionar i copiar el resultat

```python
%cd /content/lipsync-pipeline
!ffprobe -v error -show_entries format=duration -show_entries stream=width,height,r_frame_rate,nb_frames -of default=noprint_wrappers=1 data/outputs/musetalk_coarse.mp4
!cp data/outputs/musetalk_coarse.mp4 /content/drive/MyDrive/SmartDub/data/outputs/musetalk_coarse.mp4
```

## Notes de compatibilitat

MuseTalk upstream recomana Torch 2.0.1/CUDA 11.8, però PyPI `whisperx==3.8.6` requereix Torch 2.8, NumPy >=2.1, Transformers >=4.48, `faster-whisper>=1.2` i `ctranslate2>=4.5`. Per això el projecte fixa `whisperx==3.8.6` i `ctranslate2==4.8.2`; són versions estables disponibles a PyPI. La instal·lació de Colab usa `--no-deps` per als paquets que podrien reemplaçar Torch i instal·la explícitament `onnxruntime`, necessari per al VAD de faster-whisper, i `pyannote-audio`/`torchcodec`, necessaris per importar WhisperX 3.8.6. El Torch/CUDA preinstal·lat ha de passar la comprovació abans de continuar.

El demo extreu el crop de cara sencera a partir de la bounding box de `face_tracking.py`; el `mouth_roi.py` queda reservat per al compositing posterior. El wrapper converteix aquest crop facial, que pot ser rectangular, amb padding de replicació de vora a `256x256` i retorna crops BGR quadrats. No fa composició sobre el vídeo vertical ni usa la timeline de visemes per condicionar MuseTalk: el model és audio-driven. La qualitat de l’adaptació a la bounding box facial, la sincronització del nombre de frames i l’estabilitat visual requereixen validació real a la T4.
