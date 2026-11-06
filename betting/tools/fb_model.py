"""Team form (league results this season) and a Poisson model for a football fixture vs Vistabet 1X2/totals/BTTS lines."""
import json, math, re, sys, urllib.request
def get(u):
    return json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=30))
def team_ids(lg):
    t = get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{lg}/teams")
    return {x["team"]["displayName"]: x["team"]["id"] for x in t["sports"][0]["leagues"][0]["teams"]}
def results(lg, tid):
    s = get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{lg}/teams/{tid}/schedule?season=2026")
    out = []
    for e in s.get("events", []):
        c = e["competitions"][0]
        if not c.get("status", {}).get("type", {}).get("completed"): continue
        sc = {x["team"]["id"]: int(float(x["score"]["value"])) for x in c["competitors"]}
        me = sc[tid]; op = [v for k, v in sc.items() if k != tid][0]
        ha = [x["homeAway"] for x in c["competitors"] if x["team"]["id"] == tid][0]
        opn = [x["team"]["displayName"] for x in c["competitors"] if x["team"]["id"] != tid][0]
        out.append((e["date"][5:10], ha[0], opn, me, op))
    return out
def pois(k, l): return math.exp(-l) * l ** k / math.factorial(k)
for spec in sys.argv[1:]:
    lg, home, away, vb = spec.split("|")
    ids = team_ids(lg)
    hid = [v for k, v in ids.items() if home.lower() in k.lower()][0]; aid = [v for k, v in ids.items() if away.lower() in k.lower()][0]
    H = results(lg, hid); A = results(lg, aid)
    gf = lambda R: sum(r[3] for r in R) / len(R); ga = lambda R: sum(r[4] for r in R) / len(R)
    lg_avg = 1.45
    lh = gf(H) * ga(A) / lg_avg * 1.1; la = gf(A) * ga(H) / lg_avg * 0.9
    m = [[pois(i, lh) * pois(j, la) for j in range(10)] for i in range(10)]
    p1 = sum(m[i][j] for i in range(10) for j in range(10) if i > j); px = sum(m[i][i] for i in range(10)); p2 = 1 - p1 - px
    pu = {L: sum(m[i][j] for i in range(10) for j in range(10) if i + j < L) for L in (1.5, 2.5, 3.5)}
    btts = sum(m[i][j] for i in range(1, 10) for j in range(1, 10))
    txt = open(vb).read()
    def find(pat):
        x = re.search(pat, txt, re.M); return x.groups() if x else None
    mr = find(r"^Match Result 1X2 \| .+? ([\d.]+) ; X ([\d.]+) ; .+? ([\d.]+)$") or find(r"^Match Result \| .+? ([\d.]+) ; X ([\d.]+) ; .+? ([\d.]+)$") or find(r"^Result \| .+? ([\d.]+) ; X ([\d.]+) ; .+? ([\d.]+)$") or find(r"^1X2 \| .+? ([\d.]+) ; X ([\d.]+) ; .+? ([\d.]+)$")
    ou = find(r"^Total Goals \| Over 2,5 ([\d.]+) ; Under 2,5 ([\d.]+)")
    ou3 = find(r"^Total Goals \| Over 3,5 ([\d.]+) ; Under 3,5 ([\d.]+)")
    bt = find(r"^Both Teams To Score \| Yes ([\d.]+) ; No ([\d.]+)")
    print(f"\n### {home} - {away}  xG-ish {lh:.2f}-{la:.2f}")
    print(f"  {home}: GF {gf(H):.2f} GA {ga(H):.2f} | " + "; ".join(f"{r[0]} {r[1]} {r[2][:10]} {r[3]}-{r[4]}" for r in H))
    print(f"  {away}: GF {gf(A):.2f} GA {ga(A):.2f} | " + "; ".join(f"{r[0]} {r[1]} {r[2][:10]} {r[3]}-{r[4]}" for r in A))
    print(f"  model 1/X/2 {p1:.2f}/{px:.2f}/{p2:.2f}  U2.5 {pu[2.5]:.2f} U3.5 {pu[3.5]:.2f} BTTS {btts:.2f}")
    dnb = find(r"^Draw No Bet \| .+? ([\d.]+) ; .+? ([\d.]+)$")
    if dnb:
        q1 = p1 / (p1 + p2); print(f"  VB DNB {dnb} EV1 {q1*float(dnb[0])-1:+.2f} EV2 {(1-q1)*float(dnb[1])-1:+.2f}")
    if mr: print(f"  VB 1X2 {mr} EV1 {p1*float(mr[0])-1:+.2f} EVX {px*float(mr[1])-1:+.2f} EV2 {p2*float(mr[2])-1:+.2f}")
    if ou: print(f"  VB O/U2.5 {ou} EVo {(1-pu[2.5])*float(ou[0])-1:+.2f} EVu {pu[2.5]*float(ou[1])-1:+.2f}")
    if ou3: print(f"  VB O/U3.5 {ou3} EVo {(1-pu[3.5])*float(ou3[0])-1:+.2f} EVu {pu[3.5]*float(ou3[1])-1:+.2f}")
    if bt: print(f"  VB BTTS {bt} EVy {btts*float(bt[0])-1:+.2f} EVn {(1-btts)*float(bt[1])-1:+.2f}")
