"""Download tonight's NBA scoreboard and ESPN summaries of every completed game of the teams playing (cached), plus tonight's games."""
import json, os, sys, urllib.request
base, date = sys.argv[1], sys.argv[2]
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124"}
def get(u):
    return json.load(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=40))
sb = get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date}")
json.dump(sb, open(os.path.join(base, "sb.json"), "w"))
ids = set()
for e in sb["events"]:
    ids.add(e["id"])
    for c in e["competitions"][0]["competitors"]:
        s = get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{c['team']['id']}/schedule?season=2027&seasontype=2")
        for g in s.get("events", []):
            if g["competitions"][0]["status"]["type"]["completed"]:
                ids.add(g["id"])
n = 0
for i in ids:
    p = os.path.join(base, "sum", f"{i}.json")
    if i in {e["id"] for e in sb["events"]} or not os.path.exists(p):
        json.dump(get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={i}"), open(p, "w")); n += 1
print(len(sb["events"]), "games tonight;", len(ids), "summaries;", n, "downloaded")
for e in sb["events"]:
    print(e["id"], e["date"], e["shortName"])
