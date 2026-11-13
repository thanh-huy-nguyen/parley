# Using Parley

This page is the reference for the editor: what it accepts, what it returns, what the edit language covers, how it
behaves on unusual inputs, and what a session keeps.

## Input and output contract

| Item | Form | Notes |
|---|---|---|
| Image | Array or path. uint8 (H, W, 3) is used as is. Grayscale, RGBA, 16-bit and float in 0..1 are converted (see the behavior table). | Resampled to 512 x 512, bilinear. Aspect ratio is not kept. |
| Mask | bool or numeric (H, W), nonzero is foreground; or a path, where foreground is above half of the file's maximum. | The output has the same (H, W). |
| Working grid | 512 x 512 px | The network, the signed distance function and D for its programs live here. The output boundary has this resolution, whatever the input size. On non-square inputs, working pixels are anisotropic. |
| D | Equivalent diameter, sqrt(4 area / pi), in pixels | All edit amounts are fractions of D. The network floors D at 8 working pixels, and an empty mask has D = 128 working pixels. |
| Click | One (x, y) in pixels of the mask, floats allowed, origin at the top left, x to the right, y downward | One per turn. Used as a heat map with sigma 0.03 D. A list of clicks is an error. |
| Text | str, English | Truncated at 64 tokens. `text` wins if audio is also given. |
| Audio | Path (anything ffmpeg and libsndfile read), or `samples` as float or int (n) or (n, channels) with `sample_rate` | 16 kHz mono is expected; other forms are converted with a warning. |

`editor.edit(...)` returns an `EditResult`:

| Field | Type | Meaning |
|---|---|---|
| `mask` | bool (H, W) | The new mask |
| `status` | str | `edited`, `unchanged` (a program ran but no pixel changed), `accepted`, `no_input` (silence or empty transcript, no edit) or `low_confidence` (only with `min_confidence`) |
| `transcript` | str or None | What the program was read from |
| `programs` | list of dict, at most 2 | `type`, `scope`, `place`, `move_dir`, `sign`, `size`, `amount_D`, `move_D`, `confidence`. The network fills every field; only some apply: `scope`, `sign`, `size`, `amount_D` to shrink; `move_dir`, `move_D` to move; `place` to add and remove |
| `changed_fraction` | float | Share of pixels that changed |
| `warnings` | list of str | Notes on the inputs, also raised as `ParleyInputWarning` |

Other methods: `begin_session(clinician_id, dataset)`, `begin_image(image, start_mask)`, `undo()`, `end_image(final_mask, accepted, accept_text)`, `belief()`, `save_state(path)`, `load_state(path, dataset=None)`, `transcribe(wav or samples)`.

Modes: `base` reads and executes. `listener` adds a per-clinician margin belief, whose mean offsets each new start mask once. `adapter` adds a rank-4 correction of the decoders, fitted at the end of each accepted image. `listener+adapter` has both. `parley.Sam2ClickEditor` is a click-only comparator built on SAM 2.1 tiny, used in the evaluation kit and the benchmark.

## What the edit language can and cannot express

Each turn holds up to two programs, run in the order spoken.

| You can say | Program |
|---|---|
| "pull the top in a little" | shrink, top arc, small |
| "too tight all round, open it up a lot" | grow, whole contour, large |
| "shift it down and to the left" | move, one of eight directions, with a predicted vector |
| "there's a piece missing at the bottom" | add a region, side bottom |
| "get rid of that blob" plus a click | remove a region, near the click |
| "that looks good" | accept, mask unchanged |

This table shows what each kind of sentence is meant to produce. The network does not always deliver it, and the displayed program is the way to check. On seed 0, for example, "that's fine" parsed as none rather than accept, "open it up a lot" gave a small or medium grow, and "make it a bit bigger, and pull the left side in" ran as a single grow of the left side.

Size words map to bins in training: small is below 0.04 D, medium is 0.04 to 0.12 D, large is above 0.12 D. The network regresses the amount on its own, so the amount need not fall in the bin it names. The anchors are the whole contour and four arcs of the screen (top, bottom, left, right). Sides are those of the image, not of the patient. An upper-left request becomes two overlapping edits.

The language does not express smoothing or regularizing a contour; filling a hole or merging parts on request; anatomy ("leave out the muscle"); appearance ("where it is darker"); absolute lengths ("two millimeters"), since pixel spacing is not an input; or more than two edits in a turn, of which the rest are dropped. A `split` type exists in the parser, but it runs only the boundary refinement and cuts nothing. Parley edits one 2D frame and does not read DICOM or volumes.

## Behavior on unusual inputs

The core algorithm is not changed by any of these except where the last column says so. Each row has a test in `tests/test_edge_cases.py`.

| Input | Behavior | Kind |
|---|---|---|
| Empty mask | Runs. Shrink, grow and move leave it empty; an add program can create a region. | Warning |
| Full-image mask | Runs. Shrink, grow and move leave it full; a remove program can carve a region. | Warning |
| One-pixel or very small mask | Runs with D floored at 8 working pixels. A mask smaller than a working pixel can vanish on resampling, and it is then treated as empty. | Warning |
| Several components | Runs. D and the side windows use all components together; a move moves them all. Touching at a corner counts as one component. | Warning |
| Empty or blank transcript | No edit, status `no_input`. `pass_empty_transcript=True` runs the network instead. | No-op, added by this package |
| Silent audio (RMS below 1e-4) or no samples | No edit, status `no_input`; the recognizer does not run. Low noise that is not silent reaches the recognizer, which may invent words. | No-op, added by this package |
| Off-image click | Passed on as given; its heat map fades with distance. | Warning |
| List of clicks, wrong length, non-numeric, NaN | `InputError` | Error |
| Sample rate other than 16 kHz, or several channels | Converted to 16 kHz mono. | Warning |
| Samples without a rate; missing or unreadable audio file | `InputError` | Error |
| Grayscale, or gray with alpha | Repeated to three channels. | Warning |
| RGBA | Alpha dropped without compositing. | Warning |
| 16-bit or 32-bit integer image | Rescaled linearly from its own minimum and maximum to 8 bits. | Warning, added by this package |
| Float image | Must be finite and in 0..1, otherwise `InputError`. Scaled to 8 bits. | Warning or error, added by this package |
| Image and mask of different size | Both go to the working grid; clicks are read in mask pixels. | Warning |
| Long side above 4096 px | Runs. The boundary has only 512 x 512 resolution, and listener updates slow down. | Warning |
| More than 178,956,970 pixels | `ImageTooLargeError` | Error, added by this package |
| DICOM file | `InputError` with the reason | Error |
| Mask not (H, W), or bad image shape or dtype | `InputError` | Error |

For strict checking, turn warnings into errors with `warnings.simplefilter("error", parley.errors.ParleyInputWarning)`.

## Session state: what persists and what resets

A session is one clinician on one dataset. `begin_session` resets everything that was learned: the listener returns to its prior and the adapters to a fresh initialization. A new dataset is a new session, because a margin learned on one modality has no reason to carry to another. Start a session per dataset, even for the same clinician.

The `dataset` argument is only a label. Any name is accepted, it does not select a prior or change the model, and `load_state(path, dataset="x")` refuses state saved under another label.

`save_state(path)` writes the listener's log posterior, the adapter matrices and the session's identity to a `.npz` that loads without pickle. `load_state(path)` restores them exactly; the mode must match, and for adapters so must the seed. The file holds no images, masks or transcripts. The current image's turns, its reference mask and the undo history are not saved, and `begin_image` and `end_image` clear them.

`undo` restores the mask, the listener's belief and the turn list. `end_image` clears the undo history, since the accepted image has then taught what it can.

Only an accepted image teaches the adapters. The listener updates on each shrink or grow turn, and on an accepted image when `accept_text` is given and reads as an accept.


## Running the tests

```bash
pytest -m "not weights and not gpu"          # CPU, no weights, no downloads
PARLEY_WEIGHTS=/path pytest -m weights        # real models; PARLEY_TEST_DEVICE=cuda to use a GPU
```
