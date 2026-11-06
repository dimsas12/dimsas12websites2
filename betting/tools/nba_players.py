"""Per-player NBA logs for tonight's teams from ESPN summaries; writes players.json and prints a report."""
import json, glob, os, sys, statistics as st
from collections import defaultdict

base = sys.argv[1]
sb = json.load(open(os.path.join(base, "sb.json")))
tonight = {e["id"] for e in sb["events"]}
teams = {c["team"]["abbreviation"] for e in sb["events"] for c in e["competitions"][0]["competitors"]}

logs = defaultdict(list)     # (team, name) -> games
team_games = defaultdict(list)
injuries = {}
for f in sorted(glob.glob(os.path.join(base, "sum", "*.json"))):
    ev = os.path.basename(f)[:-5]
    d = json.load(open(f))
    if ev in tonight:
        for t in d.get("injuries", []):
            ab = t["team"]["abbreviation"]
            injuries[ab] = [(i["athlete"]["displayName"], i.get("status"), (i.get("details") or {}).get("type", "")) for i in t.get("injuries", [])]
        continue
    comp = d["header"]["competitions"][0]
    date = comp["date"][:10]
    sc = {c["team"]["abbreviation"]: int(c.get("score", 0) or 0) for c in comp["competitors"]}
    for c in comp["competitors"]:
        ab = c["team"]["abbreviation"]
        opp = [k for k in sc if k != ab][0]
        team_games[ab].append((date, opp, sc[ab], sc[opp], c["homeAway"]))
    for side in d.get("boxscore", {}).get("players", []):
        ab = side["team"]["abbreviation"]
        if ab not in teams:
            continue
        s = side["statistics"][0]
        names = s["names"] if "names" in s else s["labels"]
        for a in s["athletes"]:
            if not a.get("stats"):
                continue
            row = dict(zip(names, a["stats"]))
            try:
                mn = float(row["MIN"])
            except ValueError:
                continue
            if mn <= 0:
                continue
            fg3m, fg3a = (int(x) for x in row["3PT"].split("-"))
            fgm, fga = (int(x) for x in row["FG"].split("-"))
            ftm, fta = (int(x) for x in row["FT"].split("-"))
            logs[(ab, a["athlete"]["displayName"])].append(dict(
                date=date, min=mn, pts=int(row["PTS"]), reb=int(row["REB"]), ast=int(row["AST"]),
                tpm=fg3m, tpa=fg3a, fga=fga, fta=fta, stl=int(row["STL"]), blk=int(row["BLK"]),
                starter=a.get("starter", False)))

out = {}
for e in sb["events"]:
    c = e["competitions"][0]
    o = (c.get("odds") or [{}])[0]
    h = [x for x in c["competitors"] if x["homeAway"] == "home"][0]["team"]["abbreviation"]
    a = [x for x in c["competitors"] if x["homeAway"] == "away"][0]["team"]["abbreviation"]
    print(f"\n######## {a} @ {h}  {e['date']}  line {o.get('details')}  O/U {o.get('overUnder')}")
    for t in (a, h):
        g = sorted(team_games[t])
        print(f"== {t} results: {[(x[0][5:], x[4][0], x[1], f'{x[2]}-{x[3]}') for x in g]}  PF {st.mean(x[2] for x in g):.1f} PA {st.mean(x[3] for x in g):.1f}")
        if g:
            print(f"   last game {g[-1][0]} (b2b if 2026-10-27)")
        print(f"   injuries: {injuries.get(t, [])}")
        rows = sorted([(k[1], sorted(v, key=lambda x: x['date'])) for k, v in logs.items() if k[0] == t], key=lambda r: -st.mean(x['min'] for x in r[1]))
        for name, v in rows:
            mm = st.mean(x["min"] for x in v)
            if mm < 10:
                continue
            m = lambda k: st.mean(x[k] for x in v)
            out[f"{t}|{name}"] = v
            print(f"   {name:24s} gp{len(v)} min{mm:5.1f} pts{m('pts'):5.1f} reb{m('reb'):5.1f} ast{m('ast'):4.1f} 3pm{m('tpm'):4.1f} fga{m('fga'):5.1f} fta{m('fta'):4.1f} | P{[x['pts'] for x in v]} R{[x['reb'] for x in v]} A{[x['ast'] for x in v]} 3{[x['tpm'] for x in v]} M{[round(x['min']) for x in v]}")
json.dump(out, open(os.path.join(base, "players.json"), "w"))
