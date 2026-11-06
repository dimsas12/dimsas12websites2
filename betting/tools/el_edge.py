"""Join Vistabet Euroleague player props (English dump) with Euroleague box-score logs; rank edges with the NBA model."""
import json, os, re, sys, math, unicodedata, statistics as st
from collections import defaultdict
base = sys.argv[1]
sys.argv_txt = sys.argv[2:]
games = json.load(open(os.path.join(base, "el.json")))["data"]
played = [g for g in games if g["played"]]

def mins(s):
    if not s or ":" not in s: return 0.0
    m, _, sec = s.partition(":"); return int(m) + int(sec or 0) / 60
def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", "", s.replace("-", " ")).split()

logs = defaultdict(list)
for g in sorted(played, key=lambda g: g["date"]):
    b = json.load(open(os.path.join(base, "box", f"{g['gameCode']}.json")))
    for side in b["Stats"]:
        for p in side["PlayersStats"]:
            mn = mins(p["Minutes"])
            if mn == 0: continue
            logs[p["Player"].strip()].append(dict(min=mn, pts=p["Points"], reb=p["TotalRebounds"], ast=p["Assistances"], tpm=p["FieldGoalsMade3"]))
# box name "LAST, FIRST" -> token set
index = {}
for k in logs:
    last, _, first = k.partition(",")
    index[k] = (norm(last), norm(first))
def find(name):
    toks = norm(re.sub(r"\(.*?\)", "", name))
    best = None
    for k, (last, first) in index.items():
        if last and all(t in toks for t in last) and (not first or first[0] in toks):
            if best is None or len(logs[k]) > len(logs[best]): best = k
    return best

MK = {"points": "pts", "rebounds": "reb", "assists": "ast", "three-pointers made": "tpm", "total points, rebounds and assists": "pra"}
LAD = {"to score": "pts", "to record": None, "to make": "tpm"}
props = []
for path in sys.argv[2:]:
    game = os.path.basename(path)[:-4]
    for ln in open(path):
        if "|" not in ln: continue
        name, opts = [x.strip() for x in ln.split("|", 1)]
        m = re.fullmatch(r"(.+?)\s*\(\w+\)\s*:\s*(.+)", name)
        if m and m.group(2).lower() in MK:
            o = re.search(r"Over ([\d.,]+) ([\d.]+) ; Under [\d.,]+ ([\d.]+)", opts)
            if o: props.append(dict(game=game, player=m.group(1), market=MK[m.group(2).lower()], kind="ou",
                                    line=float(o.group(1).replace(",", ".")), over=float(o.group(2)), under=float(o.group(3))))
            continue
        m = re.fullmatch(r"To (score|record|make) (\d+)\+ (points|rebounds|assists|three-pointers)", name, re.I)
        if m:
            mk = {"points": "pts", "rebounds": "reb", "assists": "ast", "three-pointers": "tpm"}[m.group(3).lower()]
            for part in opts.split(" ; "):
                q = re.fullmatch(r"(.+?) ([\d.]+)", part.strip())
                if q: props.append(dict(game=game, player=q.group(1), market=mk, kind="lad", line=int(m.group(2)) - 0.5, over=float(q.group(2)), under=None))

def series(v, m):
    return [g["pts"] + g["reb"] + g["ast"] for g in v] if m == "pra" else [g[m] for g in v]
def p_over(mu, line, m):
    if m == "tpm":
        k = int(math.floor(line)); return 1 - sum(math.exp(-mu) * mu ** i / math.factorial(i) for i in range(k + 1))
    sd = {"pts": 0.30 * mu + 2.5, "reb": 0.32 * mu + 1.2, "ast": 0.34 * mu + 0.9, "pra": 0.25 * mu + 3.0}[m]
    return 0.5 * math.erfc((line - mu) / sd / math.sqrt(2))
def solve_mu(line, fair, m):
    lo, hi = 0.05, 3 * line + 20
    for _ in range(60):
        mid = (lo + hi) / 2
        if p_over(mid, line, m) < fair: lo = mid
        else: hi = mid
    return (lo + hi) / 2
anchor = {}
for p in props:
    if p["kind"] == "ou":
        fair = (1 / p["over"]) / (1 / p["over"] + 1 / p["under"])
        anchor[(norm(p["player"])[-1], p["market"])] = solve_mu(p["line"], fair, p["market"])
lad = defaultdict(list)
for p in props:
    if p["kind"] == "lad": lad[(norm(re.sub(r"\(.*?\)", "", p["player"]))[-1], p["market"])].append((p["line"], 0.92 / p["over"]))
for k, pts in lad.items():
    if k in anchor or len(pts) < 2: continue
    pts.sort()
    for (l1, q1), (l2, q2) in zip(pts, pts[1:]):
        if q1 >= 0.5 >= q2 and q1 != q2:
            anchor[k] = solve_mu(l1 + (q1 - 0.5) / (q1 - q2) * (l2 - l1), 0.5, k[1]); break
rows, miss = [], set()
for p in props:
    k = find(p["player"])
    if not k: miss.add(p["player"]); continue
    v = logs[k]; s = series(v, p["market"]); n = len(s); avg = st.mean(s)
    key = (norm(re.sub(r"\(.*?\)", "", p["player"]))[-1], p["market"])
    tgt = anchor.get(key, p["line"]); w = n / (n + 6); mu = w * avg + (1 - w) * tgt
    po = p_over(mu, p["line"], p["market"])
    ev_o = po * p["over"] - 1 if p["over"] <= 3.2 else None
    ev_u = (1 - po) * p["under"] - 1 if p["under"] else None
    rows.append(dict(p, box=k, n=n, avg=round(avg, 1), hits=f"{sum(x > p['line'] for x in s)}/{n}", mu=round(mu, 2), po=round(po, 3), ev_o=ev_o, ev_u=ev_u, log=s, mins=[round(x['min']) for x in v]))
json.dump(rows, open(os.path.join(base, "el_edges.json"), "w"))
print("props", len(props), "rows", len(rows), "missing", sorted(miss))
def best(r):
    c = [(r["ev_o"], "O", r["over"])] if r["ev_o"] is not None else []
    if r["ev_u"] is not None: c.append((r["ev_u"], "U", r["under"]))
    return max(c) if c else None
for b, r in sorted([(best(r), r) for r in rows if best(r)], key=lambda x: -x[0][0])[:45]:
    print(f"{r['game']:8s} {r['player']:30s} {r['market']:3s} {r['kind']} {b[1]}{r['line']:5} @{b[2]:.2f} avg {r['avg']:5} hits {r['hits']:4s} mu {r['mu']:6} P(o) {r['po']:.2f} EV {b[0]:+.3f} log {r['log']} min {r['mins']}")
