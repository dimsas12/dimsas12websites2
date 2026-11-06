"""Per-player Euroleague logs for today's teams, plus team scoring, from downloaded box scores."""
import json, os, sys, statistics as st
from collections import defaultdict

base = sys.argv[1]
teams = sys.argv[2].split(",")
games = json.load(open(os.path.join(base, "el.json")))["data"]
played = sorted([g for g in games if g["played"]], key=lambda g: g["date"])

def mins(s):
    if not s or s in ("DNP",):
        return 0.0
    m, _, sec = s.partition(":")
    return int(m) + int(sec or 0) / 60

logs = defaultdict(list)  # (team, player) -> list of dicts
team_pts = defaultdict(list)  # team -> list of (for, against, home)
for g in played:
    b = json.load(open(os.path.join(base, "box", f"{g['gameCode']}.json")))
    lc, rc = g["local"]["club"]["code"], g["road"]["club"]["code"]
    ls, rs = g["local"]["score"], g["road"]["score"]
    team_pts[lc].append((ls, rs, "H", rc, g["round"]))
    team_pts[rc].append((rs, ls, "A", lc, g["round"]))
    for side in b["Stats"]:
        for p in side["PlayersStats"]:
            code = p["Team"].strip()
            mn = mins(p["Minutes"])
            if mn == 0:
                continue
            logs[(code, p["Player"].strip())].append(dict(
                r=g["round"], min=mn, pts=p["Points"], reb=p["TotalRebounds"],
                ast=p["Assistances"], tpm=p["FieldGoalsMade3"], tpa=p["FieldGoalsAttempted3"],
                fga=p["FieldGoalsAttempted2"] + p["FieldGoalsAttempted3"], fta=p["FreeThrowsAttempted"]))

for t in teams:
    tp = team_pts[t]
    pf = [x[0] for x in tp]; pa = [x[1] for x in tp]
    print(f"\n=== {t}  games {len(tp)}  PF {st.mean(pf):.1f}  PA {st.mean(pa):.1f}  totals {[a+b for a,b,*_ in tp]}")
    print("   results:", [(x[4], x[2], x[3], f"{x[0]}-{x[1]}") for x in tp])
    rows = [(k[1], v) for k, v in logs.items() if k[0] == t]
    rows.sort(key=lambda kv: -st.mean(x["min"] for x in kv[1]))
    for name, v in rows:
        n = len(v)
        f = lambda key: st.mean(x[key] for x in v)
        last = v[-3:]
        print(f"  {name:28s} gp{n} min{f('min'):5.1f} pts{f('pts'):5.1f} reb{f('reb'):4.1f} ast{f('ast'):4.1f} 3pm{f('tpm'):4.1f} fga{f('fga'):4.1f} | pts {[x['pts'] for x in v]} reb {[x['reb'] for x in v]} ast {[x['ast'] for x in v]} 3pm {[x['tpm'] for x in v]} | last3min {[round(x['min']) for x in last]}")
