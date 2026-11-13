# Model card: Parley editor network

## Summary

The network reads one correction, as text, on an image and a current mask, and produces up to two edit programs that it runs on the mask's signed distance function. It has about 7.6 million parameters in use (`parley.model.ParleyNet`). The image encoder (SAM 2.1 tiny), the text encoder (all-MiniLM-L6-v2) and the recognizer (whisper-small.en) are frozen and were not trained by us. Three checkpoints exist, from seeds 0, 1 and 2. Results in the paper average over them.

## Intended use

Research on spoken correction of 2D segmentation masks, and as a subject for the evaluation kit. Research use only.

Not intended for diagnosis, treatment, or any decision about a patient. Not cleared or approved by any regulator. Not evaluated in a clinical study and not trained on recorded clinical speech; two physicians have tried the demo with a live microphone.

## Training data

Images: BUSI (breast ultrasound; normal images without a lesion were dropped, and multi-mask images were merged) and Kvasir-SEG (colonoscopy polyps). Images were split by near-duplicate clusters. Training used the training split only: BUSI 453 images and Kvasir-SEG 700 images. The copies used came from community mirrors on Hugging Face (see `THIRD_PARTY.md`).

BUSI images keep their original overlays and caliper marks, and a model could learn from them.

Language: no human speech. A generator simulates clinicians with a margin drawn from -0.04 to 0.04 D. An oracle decomposes the remaining error of a draft mask into edit programs. A language model (Qwen2.5-7B-Instruct) phrases each program. A text-to-speech system (Kokoro) renders it. whisper-small.en transcribes it, so recognition errors are part of training. In total, 200 simulated clinicians produced 9,280 dialogues.

Start masks: a mix of out-of-fold predictions of a U-Net trained per dataset and synthetic degradations of the ground truth, plus a round of states the model itself reached.

## Inputs and outputs

See the input and output contract in `docs/usage.md`.

## Results

From the paper, with speech and click, on 2D dialogues whose start mask needs correction (mean over seeds 0 to 2).
Reach is the share that gets within IoU 0.85 of the simulated clinician's target in 10 turns; worse is the share whose
last mask has a lower IoU than the start.

| Simulated clinicians | Dialogues | Reach | Mean gain in IoU | Improved | Worse |
|---|---:|---:|---:|---:|---:|
| Simulator built with the editor (validation images) | 356 | 37.5% | +0.140 | 88.5% | 8.7% |
| Separately written personas (validation images) | 593 | 47.9% | +0.154 | 93.4% | 4.7% |
| Separately authored simulator (test images) | 376 | 17.7% | +0.026 | 52.7% | 43.1% |

The model was selected on the first group, so those dialogues are unseen only in training. On the separately authored
simulator, which changes the language, voices and images together, the gain almost vanishes and about four in ten
masks end worse than they started. With speech alone there, the mean gain is negative (-0.034). SAM 2 given the same
simulator's clicks reached the target more often. In a research-code extension to three volumetric datasets, editing
one slice per turn, no speech editor improved volumes when edits were propagated by SAM 2.

## Known failures and limits

- Masks can get worse: the program is executed even when it is wrong. See the results above once filled in.
- Region quality on add and remove depends on the image; nothing forces the region to be what the speaker meant.
- The model does not know when an image is outside what it was trained on.
- The listener assumes one scalar margin per clinician and dataset, and it uses the lesion head's mask as its reference, so errors in that mask pass into the belief.
- See "Limitations" in `README.md` and "What the edit language can and cannot express" in `docs/usage.md`.

## Versions and integrity

The SHA-256 of each checkpoint and the commit of each frozen model are in `src/parley/weights_manifest.json`. Each checkpoint also stores about 0.45 million parameters that inference does not use; they are dropped at load time.

## License of the weights

CC BY-NC 4.0, for non-commercial research use (see `WEIGHTS_LICENSE.md`). Kvasir-SEG's terms restrict the dataset to research and educational use and forbid commercial use without written permission. See `THIRD_PARTY.md`.
