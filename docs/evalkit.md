# The evaluation kit

`parley.evalkit` tests any editor with four methods (`begin_session`, `begin_image`, `step`, `end_image`) through any simulator that implements `SimulatedClinician` (`target`, `accept`, `next_utterance`, `click`).

```python
from parley import ParleyEditor
from parley.evalkit import Case, ParleyEvalAdapter, run_study, summarize, summary_markdown
from parley.evalkit.toy import ToyClinician      # an example only

records = run_study(ParleyEvalAdapter(ParleyEditor()), [ToyClinician("a", margin=0.02)], {"my_data": [Case("id1", "my_data", rgb, start, truth)]})
print(summary_markdown({"parley": summarize(records)}))
```

The outcomes are defined in `parley/evalkit/outcomes.py`:

- Reach: the IoU with the target reaches a threshold at any of T turns (default 0.85 and 10). The mask the editor shows first counts as turn 0. `needs_correction=True` restricts a summary to dialogues whose given start mask is below the threshold, whatever the editor does to that mask before the first turn.
- Gain: the mean IoU change from the given start mask to the last mask. `gain_stop` uses the first mask at or above the threshold, else the last.
- Improved and worse: the shares of dialogues with positive and negative gain.
- Simulated time: 0.4 s per word, 0.5 s for the recognizer on turns with speech, 2.0 s per click and 4.0 s per review. These are assumed costs; change them with `TimeCosts`. Time spent (`time_s`) is the turns taken, plus, for a clinician who gives up, the remaining turns up to T at that dialogue's mean cost per turn. Time to reach (`time_reach`) stops at the first mask at or above the threshold, and charges a dialogue that never reaches it the remaining turns, however it ended.
- Intervals: a two-way clustered bootstrap that resamples clinicians and images independently.

The kit does not compute agreement on the edit type (the simulator interface carries no intended type), does not exclude non-edit turns from time, does not separate later cases, and has no floor at a clinician's mean cost per turn, since it charges each dialogue on its own. A dialogue with no turns is charged the review alone per turn, which is a lower bound. You supply the simulator. Turns reach the editor as written text unless you give `ParleyEvalAdapter` a `wav_for` that supplies the speech.

`ToyClinician` is written only to show the interface. Numbers from it are not an evaluation. `parley.evalkit.perturb` adds word errors and click jitter for robustness runs.

## Running the example

```bash
python examples/evalkit_toy_run.py --device cpu --out toy_run.json
python examples/evalkit_toy_run.py --images my/images --masks my/masks --starts my/starts --out toy_run.json
```

Images, reference masks and start masks are paired by file name. The script runs the editor in `base` and `listener`
modes and prints the reporting table.
