# MuseTalk en Colab

## Cel·la 1: muntar Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Cel·la 2: clonar el repositori amb el submòdul

```python
%cd /content
REPO_URL = "https://github.com/EL_TEU_USUARI/lipsync-pipeline.git"  # Canvia només aquesta URL.
!git clone --recurse-submodules "$REPO_URL" lipsync-pipeline
%cd /content/lipsync-pipeline
!git submodule update --init --recursive
```

## Cel·la 3: comprovar GPU i FFmpeg

```python
!nvidia-smi
!ffmpeg -version | head -n 2
```

## Cel·la 4: instal·lar dependències sense substituir Torch de Colab

```python
%cd /content/lipsync-pipeline
# Colab ja porta torch/torchvision/torchaudio compilats per la seva CUDA.
# No instal·lem requirements.txt complet ni la versió Torch de MuseTalk.
!pip install -q --no-deps \
  "diffusers==0.30.2" "accelerate==0.28.0" \
  "transformers==4.47.1" "huggingface_hub==0.26.5" \
  "whisperx==3.3.1" "g2p_en==2.1.0" \
  "opencv-python==4.9.0.80" "soundfile==0.12.1" \
  "librosa==0.10.2.post1" "einops==0.8.1" \
  "omegaconf" "ffmpeg-python"

# Dependències Python sense una instal·lació pròpia de Torch.
!pip install -q \
  "faster-whisper==1.1.0" "ctranslate2==4.3.1" "av" \
  "pandas" "scipy" "tqdm" "pyyaml" "sentencepiece" \
  "inflect==7.5.0" "distance==0.1.3"

# Verifica que el Torch preinstal·lat continua sent el que veu Colab.
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
assert torch.cuda.is_available()
```

## Cel·la 5: descarregar checkpoints MuseTalk

```python
%cd /content/lipsync-pipeline
!pip install -q --no-deps "huggingface_hub[cli]" gdown
!mkdir -p models/musetalk models/musetalkV15 models/sd-vae models/whisper

!huggingface-cli download TMElyralab/MuseTalk \
  --local-dir models \
  --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth"
!huggingface-cli download stabilityai/sd-vae-ft-mse \
  --local-dir models/sd-vae \
  --include "config.json" "diffusion_pytorch_model.bin"
!huggingface-cli download openai/whisper-tiny \
  --local-dir models/whisper \
  --include "config.json" "pytorch_model.bin" "preprocessor_config.json"
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

MuseTalk upstream recomana Torch 2.0.1/CUDA 11.8 i declara `numpy==1.23.5`, `transformers==4.39.2` i `librosa==0.11.0`. Aquest repositori conserva el Torch/CUDA de Colab i prioritza els seus pins: `numpy==1.26.4`, `transformers==4.47.1`, WhisperX `3.3.1` i `librosa==0.10.2.post1`. Les instal·lacions de `diffusers`, `accelerate`, `transformers`, `whisperx` i `g2p_en` fan servir `--no-deps` per evitar que pip substitueixi Torch; si una cel·la detecta un conflicte binari, reinicia el runtime abans de continuar.

El demo extreu el crop de cara sencera a partir de la bounding box de `face_tracking.py`; el `mouth_roi.py` queda reservat per al compositing posterior. El wrapper converteix aquest crop facial, que pot ser rectangular, amb padding de replicació de vora a `256x256` i retorna crops BGR quadrats. No fa composició sobre el vídeo vertical ni usa la timeline de visemes per condicionar MuseTalk: el model és audio-driven. La qualitat de l’adaptació a la bounding box facial, la sincronització del nombre de frames i l’estabilitat visual requereixen validació real a la T4.
