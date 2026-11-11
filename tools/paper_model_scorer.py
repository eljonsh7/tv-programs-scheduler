"""Independent scorer written directly from the problem definition in the PATAT 2026 paper
(Smart TV Scheduling in Public Venues). It does not use validator.js, so it can be used to
cross-check the validator.

Usage (from the repository root):
    python3 tools/paper_model_scorer.py [instance ...]

Default instances: croatia kosovo netherlands uk_synthetic. It scores outputs/ilp/<instance>.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def score(inst, sched):
    O, E, D, R = inst["opening_time"], inst["closing_time"], inst["min_duration"], inst["max_consecutive_genre"]
    S, T = inst["switch_penalty"], inst["termination_penalty"]
    P = {}
    for ch in inst["channels"]:
        for p in ch["programs"]:
            P[(ch["channel_id"], p["program_id"])] = dict(c=ch["channel_id"], a=p["start"], b=p["end"], g=p["genre"], v=p["score"])
    viol, seen = [], set()
    base = bonus = late = early = sw = 0
    prev, streak, pg = None, 0, None
    for (c, pid, s, e) in sched:
        k = (c, pid)
        p = P.get(k)
        if p is None:
            viol.append(f"unknown {k}")
            continue
        if k in seen:
            viol.append(f"program {k} scheduled twice")
        seen.add(k)
        d, delta = e - s, p["b"] - p["a"]
        if s < max(O, p["a"]) or e > min(E, p["b"]):
            viol.append(f"{pid}: outside [max(O,a), min(E,b)] s={s} e={e} a={p['a']} b={p['b']}")
        if d < min(D, delta):
            viol.append(f"{pid}: duration {d} < min(D,delta)={min(D, delta)}")
        for blk in inst["priority_blocks"]:
            if c not in blk["allowed_channels"] and not (e <= blk["start"] or s >= blk["end"]):
                viol.append(f"{pid}: priority block {blk['start']}-{blk['end']}")
        if prev and s < prev[1]:
            viol.append(f"{pid}: overlap with previous program")
        streak = streak + 1 if p["g"] == pg else 1
        pg = p["g"]
        if streak > R:
            viol.append(f"{pid}: genre streak {streak} > {R}")
        if prev and prev[2] != c:
            sw += 1
        base += p["v"]
        late += s > p["a"]
        early += e < p["b"]
        for w in inst["time_preferences"]:
            if w["preferred_genre"] == p["g"]:
                cond = s <= w["end"] - D and e >= w["start"] + D
                if cond and (min(p["b"], w["end"]) - max(p["a"], w["start"]) >= D):
                    bonus += w["bonus"]
        prev = (s, e, c)
    total = base + bonus - S * sw - T * (late + early)
    return dict(total=total, base=base, bonus=bonus, sw=sw, late=late, early=early, viol=viol)


if __name__ == "__main__":
    names = sys.argv[1:] or ["croatia", "kosovo", "netherlands", "uk_synthetic"]
    for n in names:
        inst = json.load(open(ROOT / "inputs" / f"{n}.json"))
        out = json.load(open(ROOT / "outputs" / "ilp" / f"{n}.json"))["scheduled_programs"]
        r = score(inst, [(x["channel_id"], x["program_id"], x["start"], x["end"]) for x in out])
        print(f"{n:13} paper-model total={r['total']:5} base={r['base']} bonus={r['bonus']} sw={r['sw']} late={r['late']} early={r['early']} violations={len(r['viol'])}")
        for v in r["viol"][:4]:
            print("     !", v)
