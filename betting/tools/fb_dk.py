"""Football: ESPN league form -> Poisson, blended 1:2 with DraftKings no-vig probabilities; EV for 1X2 and O/U."""
import json, math, sys, urllib.request
def get(u): return json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=30))
def dec(a): a = float(a); return round(1 + (a/100 if a > 0 else 100/-a), 3)
def pois(k, l): return math.exp(-l)*l**k/math.factorial(k)
for f in sys.argv[1:]:
    d = json.load(open(f)); c = d['header']['competitions'][0]; lg = d['header']['league']['slug'] if 'slug' in d['header']['league'] else None
    import os; lg = os.path.basename(f).split('|')[0]
    H = [x for x in c['competitors'] if x['homeAway']=='home'][0]['team']; A = [x for x in c['competitors'] if x['homeAway']=='away'][0]['team']
    o = d['pickcenter'][0]
    def res(tid):
        s = get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{lg}/teams/{tid}/schedule?season=2026"); out = []
        for e in s.get('events', []):
            cc = e['competitions'][0]
            if not cc.get('status',{}).get('type',{}).get('completed'): continue
            sc = {x['team']['id']: int(float(x['score']['value'])) for x in cc['competitors']}
            ha = [x['homeAway'] for x in cc['competitors'] if x['team']['id']==tid][0]
            opn = [x['team']['displayName'] for x in cc['competitors'] if x['team']['id']!=tid][0]
            out.append((e['date'][5:10], ha[0], opn[:12], sc[tid], [v for k,v in sc.items() if k!=tid][0]))
        return out
    rh, ra = res(H['id']), res(A['id'])
    gf = lambda R: sum(r[3] for r in R)/len(R); ga = lambda R: sum(r[4] for r in R)/len(R)
    lh = gf(rh)*ga(ra)/1.45*1.1; la = gf(ra)*ga(rh)/1.45*0.9
    m = [[pois(i,lh)*pois(j,la) for j in range(10)] for i in range(10)]
    p1 = sum(m[i][j] for i in range(10) for j in range(10) if i>j); px = sum(m[i][i] for i in range(10)); p2 = 1-p1-px
    line = float(o['overUnder']); pu = sum(m[i][j] for i in range(10) for j in range(10) if i+j < line)
    ml = [dec(o['homeTeamOdds']['moneyLine']), dec(o['drawOdds']['moneyLine']), dec(o['awayTeamOdds']['moneyLine'])]
    s = sum(1/x for x in ml); fair = [1/x/s for x in ml]
    ou = [dec(o['overOdds']), dec(o['underOdds'])]; so = sum(1/x for x in ou); fo = 1/ou[0]/so
    bl = [(pm + 2*fm)/3 for pm, fm in zip((p1, px, p2), fair)]; bo = ((1-pu) + 2*fo)/3
    print(f"\n### {H['displayName']} - {A['displayName']} ({lg})  λ {lh:.2f}-{la:.2f}")
    print(f"  {H['displayName']}: GF {gf(rh):.2f} GA {ga(rh):.2f} | " + '; '.join(f"{r[0]} {r[1]} {r[2]} {r[3]}-{r[4]}" for r in rh[:8]))
    print(f"  {A['displayName']}: GF {gf(ra):.2f} GA {ga(ra):.2f} | " + '; '.join(f"{r[0]} {r[1]} {r[2]} {r[3]}-{r[4]}" for r in ra[:8]))
    for lab, p, odd in (('1', bl[0], ml[0]), ('X', bl[1], ml[1]), ('2', bl[2], ml[2]), (f'O{line}', bo, ou[0]), (f'U{line}*', 1-bo, ou[1])):
        ev = p*odd-1; flag = ' <== in range' if 1.65 <= odd <= 2.5 and ev > 0.04 else ''
        print(f"  DK {lab:6s} {odd:5.2f}  p {p:.2f}  EV {ev:+.2f}{flag}")
