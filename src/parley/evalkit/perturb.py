"""Perturbations for robustness runs: errors in transcripts and in clicks."""


def inject_word_errors(text, wer, rng, vocabulary=("the", "and", "a", "it", "that", "there", "this", "so", "in", "on")):
    """Apply substitutions, deletions and insertions in equal parts, each word at probability `wer`. The word error rate
    of the output against `text` is about `wer`.
    """
    if not 0 <= wer <= 1:
        raise ValueError("wer must be in 0..1")
    out = []
    for w in text.split():
        if rng.random() < wer:
            kind = rng.integers(3)
            if kind == 0:
                out.append(str(rng.choice(vocabulary)))
            elif kind == 2:
                out += [w, str(rng.choice(vocabulary))]
        else:
            out.append(w)
    return " ".join(out)


def jitter_click(click, sd_px, rng, drop=0.0):
    """Gaussian jitter of a click in pixels. With probability `drop` the click is lost."""
    if click is None or rng.random() < drop:
        return None
    return (float(click[0] + rng.normal(0, sd_px)), float(click[1] + rng.normal(0, sd_px)))
