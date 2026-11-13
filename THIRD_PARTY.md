# Third-party terms

Checked on 2026-11-04. This is an engineering summary and not legal advice.

## Terms that shape the license of the weights

1. Kvasir-SEG. The dataset's terms restrict its use to research and educational purposes, and they forbid commercial use without prior written permission. The checkpoints were trained on it. The terms do not mention trained models, so whether they reach the weights is unclear. For this reason the weights are released for non-commercial research use only (CC BY-NC 4.0). Publications must cite the dataset paper.
2. Dataset copies. Training used community mirrors on Hugging Face, because the official downloads failed from our machine. The Kvasir-SEG mirror states no license, and the BUSI mirror's CC BY 4.0 tag is the uploader's own statement. Terms should be taken from the original sources, and the files compared with the originals if possible.
3. BUSI. Three sources disagree on the form of the terms: the data article is CC BY 4.0, a widely used Kaggle copy says CC0, and the original Cairo University page could not be read directly (see below). None found restricts trained weights. We recommend treating it as CC BY 4.0 with attribution until confirmed.

Nothing found in SAM 2.1, Whisper or MiniLM restricts distributing weights we trained on their features. Our checkpoints do not contain their weights; they are downloaded separately from their own repositories.

## Frozen models

| Component | Used as | Commit | License | Checked at |
|---|---|---|---|---|
| SAM 2.1 Hiera tiny (facebook/sam2.1-hiera-tiny) | Image encoder; click comparator | de431c4 | Apache-2.0 | Hugging Face model metadata; LICENSE in github.com/facebookresearch/sam2 (Apache-2.0). That repository also has a BSD-3-Clause LICENSE_cctorch for a component we do not use. |
| whisper-small.en (openai/whisper-small.en) | Recognizer | e872752 | Apache-2.0 on the Hugging Face repository; the original code and weights at github.com/openai/whisper are MIT | Hugging Face model metadata; LICENSE in github.com/openai/whisper |
| all-MiniLM-L6-v2 (sentence-transformers) | Text encoder | 1110a24 | Apache-2.0 | Hugging Face model metadata |

## Datasets used to train the weights

No dataset is distributed with this package.

| Dataset | License or terms | Source checked | Attribution |
|---|---|---|---|
| Kvasir-SEG | "restricted for research and educational purposes"; "commercial purposes is forbidden without prior written permission"; a reference to the dataset paper is required | datasets.simula.no/kvasir-seg, read through a web.archive.org snapshot from 2024, because the live site failed TLS verification from our machine. | Jha et al., Kvasir-SEG: A Segmented Polyp Dataset, MMM 2020 |
| BUSI | Data article CC BY 4.0 (Crossref record of DOI 10.1016/j.dib.2019.104863). Kaggle copy aryashah2k/breast-ultrasound-images-dataset says CC0, which is not the owners' statement. | Crossref; an archived Kaggle page. The live Cairo University page is behind a robot check. Its 2022 archived copy asks for a citation of the data article and states no license. | Al-Dhabyani et al., Dataset of breast ultrasound images, Data in Brief 2020 |

## Tools used only to make the training data

These generated training language and audio. Neither is a dependency of the package, and neither ships with it.

| Tool | License | Checked at |
|---|---|---|
| Qwen2.5-7B-Instruct | Apache-2.0 | Hugging Face model metadata |
| Kokoro-82M | Apache-2.0 | Hugging Face model metadata |

The version used in training and the terms of individual voices were not rechecked here.

## Python dependencies

Read from the installed METADATA by `scripts/collect_licenses.py`. All are pinned or locked; see `requirements-lock.txt`.

Points to notice:

- The CUDA build of torch pulls NVIDIA libraries (nvidia-cublas, nvidia-cudnn, nvidia-nccl and others) under NVIDIA's proprietary license. They come from pip on the user's machine. A container image built with them redistributes them, so check NVIDIA's terms before publishing such an image. The CPU image avoids them.
- The Dockerfile installs ffmpeg and libsndfile from Debian. ffmpeg's license depends on the build (LGPL or GPL), and opencv-python-headless and soundfile wheels bundle their own native libraries. Check these before publishing an image.
- certifi and parts of orjson and tqdm are MPL-2.0, which applies only to those files.
- Our package imports these libraries and does not copy their code.

