"""Rank DraftKings prop sides (from scoresandodds) against ESPN game logs with the shrunk-mean model."""
import json, math, sys, unicodedata, statistics as st
players = json.load(open(sys.argv[1])); props = json.load(open(sys.argv[2])); games = sys.argv[3].split(',')
def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return s.replace(".", "").replace("'", "").replace("-", " ").replace(" jr", "").replace(" iii", "").replace(" ii", "").strip()
by = {norm(k.split('|')[1]): (k.split('|')[0], v) for k, v in players.items()}
F = {"pts": lambda g: g["pts"], "reb": lambda g: g["reb"], "ast": lambda g: g["ast"], "tpm": lambda g: g["tpm"],
     "pra": lambda g: g["pts"]+g["reb"]+g["ast"], "pr": lambda g: g["pts"]+g["reb"], "pa": lambda g: g["pts"]+g["ast"], "ra": lambda g: g["reb"]+g["ast"]}
def p_over(mu, line, m):
    if m == "tpm":
        k = int(math.floor(line)); return 1 - sum(math.exp(-mu)*mu**i/math.factorial(i) for i in range(k+1))
    sd = {"pts": 0.28*mu+2.5, "reb": 0.30*mu+1.3, "ast": 0.32*mu+1.0, "pra": 0.22*mu+3.5, "pr": 0.24*mu+3.0, "pa": 0.24*mu+3.0, "ra": 0.28*mu+1.5}[m]
    return 0.5*math.erfc((line-mu)/sd/math.sqrt(2))
rows = []
for p in props:
    if not any(g in p['game'] for g in games): continue
    hit = by.get(norm(p['player']))
    if not hit: rows.append(dict(p, n=0)); continue
    team, v = hit; s = [F[p['market']](g) for g in v]; n = len(s); avg = st.mean(s)
    w = n/(n+6); mu = w*avg + (1-w)*p['line']
    po = p_over(mu, p['line'], p['market']); ps = po if p['side'] == 'o' else 1-po
    hits = sum((x > p['line']) if p['side']=='o' else (x < p['line']) for x in s)
    rows.append(dict(p, team=team, n=n, avg=round(avg,1), mu=round(mu,2), p=round(ps,3), ev=round(ps*p['dec']-1,3), hits=f"{hits}/{n}", log=s, mins=[round(g['min']) for g in v]))
json.dump(rows, open('dk/dk_edges.json','w'))
print('rows', len(rows), 'unmatched', [r['player'] for r in rows if not r.get('n')])
for r in sorted([r for r in rows if r.get('n')], key=lambda r: -r['ev'])[:40]:
    print(f"{r['game']:11s} {r['player']:22s} {r['market']:3s} {r['side']}{r['line']:5} {r['american']:+d} ({r['dec']}) avg {r['avg']:5} hits {r['hits']:5s} mu {r['mu']:6} p {r['p']:.2f} EV {r['ev']:+.3f} {r['log']} min {r['mins']}")
