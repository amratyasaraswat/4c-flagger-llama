#!/usr/bin/env python
"""evaluate_4c.py  – v2  (text‑based matching)

Keeps the same CLI but now treats a span as the tuple
    (TYPE, pure_text)
where TYPE ∈ {Red Flag, Probe, Factor, CarePlan} and *pure_text* is the
verbatim quoted string lower‑cased and stripped.  Character offsets are
ignored – this is robust when your extractor and the gold annotator use
slightly different cleaning rules.

If you want *exact* offset matching again, flip the `KEY_FN`.
"""

import re, json, argparse, pathlib, collections, unicodedata

def normalise(s: str) -> str:
    """Case‑folding + collapse whitespace for fuzzy equality."""
    return " ".join(s.casefold().split())

LINE_RX = re.compile(
    r'^\uFEFF?'                     # optional BOM
    r'(Red\s+Flag|Probe|Factor|CarePlan)'
    r'\s*'                          # any spaces
    r'(?:<[^>]+>)?'                  # optional category we now ignore
    r'\s*\([^)]*\):\s*"(.*)"$')

KEY_FN = lambda t, txt: (t, normalise(txt))


def load(path):
    keys = collections.Counter()
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        m = LINE_RX.match(line)
        if not m:
            continue
        typ, txt = m.groups()
        keys[KEY_FN(typ, txt)] += 1
    return keys


def prf(tp, fp, fn):
    prec = tp / (tp + fp) if tp else 0.0
    rec  = tp / (tp + fn) if tp else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec and rec else 0.0
    return prec, rec, f1


def score(pred_dir, gold_dir, skip_missing=False):
    pred_dir, gold_dir = pathlib.Path(pred_dir), pathlib.Path(gold_dir)
    tp = fp = fn = 0
    for g_path in gold_dir.glob("*.txt"):
        p_path = pred_dir / g_path.name
        if not p_path.exists():
            if skip_missing:
                continue
            fn += sum(load(g_path).values())
            continue

        g_ctr = load(g_path)
        p_ctr = load(p_path)

        # True positives = min count in both
        for k in set(g_ctr) | set(p_ctr):
            g_n, p_n = g_ctr[k], p_ctr[k]
            tp += min(g_n, p_n)
            fp += max(0, p_n - g_n)
            fn += max(0, g_n - p_n)
    return prf(tp, fp, fn)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gold_dir", required=True)
    ap.add_argument("--report",  default=None)
    ap.add_argument("--skip_missing_pred", action="store_true",
                    help="ignore gold files that have no corresponding prediction")
    args = ap.parse_args()

    prec, rec, f1 = score(args.pred_dir, args.gold_dir, args.skip_missing_pred)
    out = {"precision": prec, "recall": rec, "f1": f1}
    print(json.dumps(out, indent=2))
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(out, indent=2))
