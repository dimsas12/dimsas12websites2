import json,sys,re
from collections import defaultdict
d=json.load(open(sys.argv[1]+'/sharpe.json')); R=d['rows']
def dec(a): return 1+(a/100 if a>0 else 100/-a)
def mkt(p):
    p=p.lower()
    side='over' if p.startswith('o') and re.match(r'o\d',p) else 'under' if re.match(r'u\d',p) else 'other'
    m=re.sub(r'^[ou][\d.]+\s*','',p).strip()
    return side,m
def summ(lab,rs):
    s=[r for r in rs if r['res'] in('W','L')]; w=sum(r['res']=='W' for r in s); n=len(s)
    u=sum(r['won'] or 0 for r in rs); risk=sum(1 for r in s)
    avgo=sum(r['odds'] for r in s if r['odds'])/max(1,len([r for r in s if r['odds']]))
    be=sum(1/dec(r['odds']) for r in s if r['odds'])/max(1,n)
    print(f"{lab:38s} n={n:3d} {w}-{n-w} hit {w/max(1,n):.0%} breakeven {be:.0%} units {u:+.2f} ROI {u/max(1,risk):+.0%} avg odds {avgo:+.0f}")
print('results', defaultdict(int,{k:sum(r['res']==k for r in R) for k in set(r['res'] for r in R)}))
summ('ALL',R)
for mo in ('2026-09','2026-10'): summ('month '+mo,[r for r in R if r['date'].startswith(mo)])
g=defaultdict(list)
for r in R: g[r['sport']].append(r)
for k,v in sorted(g.items(),key=lambda kv:-len(kv[1])): summ('sport '+k,v)
g=defaultdict(list)
for r in R: g[mkt(r['prop'])[0]].append(r)
for k,v in g.items(): summ('side '+k,v)
g=defaultdict(list)
for r in R: s,m=mkt(r['prop']); g[r['sport']+' | '+m.title()].append(r)
for k,v in sorted(g.items(),key=lambda kv:-len(kv[1])):
    if len(v)>=4: summ(k,v)
g=defaultdict(list)
for r in R:
    o=r['odds'] or 0
    b='<=-150' if o<=-150 else '-149..-125' if o<=-125 else '-124..-110' if o<=-110 else '-109..+100' if o<=100 else '>+100'
    g[b].append(r)
for k in ['<=-150','-149..-125','-124..-110','-109..+100','>+100']: summ('odds '+k,g[k])
# picks per day
pd=defaultdict(int)
for r in R: pd[r['date']]+=1
print('days',len(pd),'picks/day avg',round(len(R)/len(pd),1),'max',max(pd.values()))
print('stake sizes',set(r['risk'] for r in R))
