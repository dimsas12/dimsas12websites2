import json,sys,re,html
from collections import defaultdict
S=sys.argv[1]
R=json.load(open(S+'/sharpe.json'))['rows']
NB=json.load(open(S+'/shp/nba_analysis.json'))
FT={(a['r']['date'],a['r']['who'],a['r']['prop']):a for a in json.load(open(S+'/shp/nba_feat.json'))}
def dec(o): return 1+(o/100 if o>0 else 100/-o)
def e(s): return html.escape(str(s))
def stats(v):
    s=[x for x in v if x['res'] in('W','L')]; w=sum(x['res']=='W' for x in s); n=len(s); u=sum(x['won'] or 0 for x in v)
    be=sum(1/dec(x['odds']) for x in s if x['odds'])/max(1,n)
    return n,w,u,be
def row(lab,v,note=''):
    n,w,u,be=stats(v)
    cls='pos' if u>0 else 'neg'
    return f"<tr><td>{e(lab)}</td><td class=num>{n}</td><td class=num>{w}-{n-w}</td><td class=num>{w/max(1,n):.0%}</td><td class=num>{be:.0%}</td><td class='num {cls}'>{u:+.2f}</td><td class='num {cls}'>{u/max(1,n):+.0%}</td><td>{note}</td></tr>"
TH="<tr><th>Group</th><th>Bets</th><th>W-L</th><th>Hit</th><th>Break-even</th><th>Units</th><th>ROI</th><th>Read</th></tr>"
def mk(p):
    m=re.match(r'([ou])[\d.]+\s*(.+)',p,re.I)
    return (m.group(1).lower(),m.group(2).strip().title()) if m else ('-',p.strip().title())
# sport table
sp=defaultdict(list)
for x in R: sp[x['sport']].append(x)
SPN={'NFL':'Core volume. Receiving yards carry it.','NBA':'Thin edge. Points props lose, 3PM and assists win.','MLB':'Best market: pitcher outs (16-4).','Tennis':'Moneyline favourites, 20-7.','WNBA':'Playmaker assists/RA.','CFB':'Rushing yards lose.','Soccer':'Short-priced BTTS/team totals, flat.','NHL':'Small sample.','UFC':'Small sample.'}
sport_rows=''.join(row(k,v,SPN.get(k,'')) for k,v in sorted(sp.items(),key=lambda kv:-len(kv[1])))
mg=defaultdict(list)
for x in R: s,m=mk(x['prop']); mg[x['sport']+' · '+m].append(x)
mrows=''.join(row(k,v) for k,v in sorted(mg.items(),key=lambda kv:-(stats(kv[1])[2])) if len(v)>=5)
ob=defaultdict(list)
for x in R:
    o=x['odds'] or 0
    b='−150 or shorter (≤1.67)' if o<=-150 else '−149 to −125 (1.67–1.80)' if o<=-125 else '−124 to −110 (1.81–1.91)' if o<=-110 else '−109 to +100 (1.92–2.00)' if o<=100 else 'plus money (>2.00)'
    ob[b].append(x)
orows=''.join(row(k,ob[k]) for k in ['−150 or shorter (≤1.67)','−149 to −125 (1.67–1.80)','−124 to −110 (1.81–1.91)','−109 to +100 (1.92–2.00)','plus money (>2.00)'])
sd=defaultdict(list)
for x in R: sd={'o':'Overs','u':'Unders','-':'Moneyline / BTTS / other'}[mk(x['prop'])[0]] and sd; 
sd=defaultdict(list)
for x in R: sd[{'o':'Overs','u':'Unders','-':'Moneyline / BTTS / other'}[mk(x['prop'])[0]]].append(x)
srows=''.join(row(k,v) for k,v in sd.items())
# NBA categories
cat=defaultdict(list)
for a in FT.values(): cat[a['cat']].append(a['r'])
CN={'opening week: last-season baseline only':'Lines set off last season + preseason; he trusted last season.','consistent (both >=50%)':'Steady players, line near baseline.','rookie/no history':'Rookies with 2+ games of data.','FOLLOW new role: hot start, weak last season':'Chased a new role after a hot start.','BUY-LOW: weak start, strong last season':'Faded the slow start, trusted last season.'}
crows=''.join(row(k,v,CN.get(k,'')) for k,v in sorted(cat.items(),key=lambda kv:-stats(kv[1])[2]))
# NBA per-pick table
def hs(h): return f"{h['hit']}/{h['n']} · {h['avg']:g}" if h else '—'
nrows=[]
for a in NB:
    r=a['r']; res=r['res']; cls={'W':'pos','L':'neg'}.get(res,'')
    if 'line' not in a:
        nrows.append(f"<tr><td>{r['date'][5:]}</td><td>{e(r['who'])}</td><td>{e(r['prop'])}</td><td class=num>{r['odds']:+.0f}</td><td colspan=6>{e(a.get('note',''))}</td><td class='{cls}'>{res}</td></tr>"); continue
    f=FT.get((r['date'],r['who'],r['prop']),{})
    mgv=f.get('mg'); mgs=f"{mgv:+.0%}" if mgv is not None else '—'
    cur=a['cur']; last=', '.join(f"{v:g}" for v in (cur['vals'][-3:] if cur else []))
    out=', '.join(a.get('out_mates') or [])
    act=a.get('actual'); acts=f"{act:g} ({a.get('gmin') or 0:g}m)" if act is not None else 'DNP'
    nrows.append(f"<tr><td>{r['date'][5:]}</td><td>{e(a['name'])}</td><td>{e(r['prop'])}</td><td class=num>{r['odds']:+.0f}</td><td class=num>{hs(a['ls'])}</td><td class=num>{hs(a['ls15'])}</td><td class=num>{hs(cur)}{' · ['+last+']' if last else ''}</td><td class=num>{mgs}</td><td>{e(f.get('cat','—'))}{' · out: '+e(out) if out else ''}</td><td class=num>{acts}</td><td class='{cls}'>{res}</td></tr>")
# all picks by sport
allh=[]
for k,v in sorted(sp.items(),key=lambda kv:-len(kv[1])):
    rr=''.join(f"<tr><td>{x['date'][5:]}</td><td>{e(x['who'])}</td><td>{e(x['prop'])}</td><td class=num>{(x['odds'] or 0):+.0f}</td><td class=num>{dec(x['odds']):.2f}</td><td class='{ {'W':'pos','L':'neg'}.get(x['res'],'') }'>{x['res']}</td><td class=num>{(x['won'] or 0):+.2f}</td></tr>" for x in v)
    n,w,u,be=stats(v)
    allh.append(f"<details><summary>{e(k)} · {n} bets · {w}-{n-w} · {u:+.2f}u</summary><div class=tw><table><tr><th>Date</th><th>Player/Team</th><th>Prop</th><th>Odds</th><th>Dec</th><th>Res</th><th>Units</th></tr>{rr}</table></div></details>")
n,w,u,be=stats(R)
tpl=open(S+'/sharpe_tools/tpl2.html').read()
out=(tpl.replace('{{N}}',str(n)).replace('{{WL}}',f"{w}-{n-w}").replace('{{HIT}}',f"{w/n:.0%}").replace('{{BE}}',f"{be:.0%}").replace('{{U}}',f"{u:+.1f}").replace('{{ROI}}',f"{u/n:+.0%}")
 .replace('{{TH}}',TH).replace('{{SPORT}}',sport_rows).replace('{{MKT}}',mrows).replace('{{ODDS}}',orows).replace('{{SIDE}}',srows).replace('{{CAT}}',crows)
 .replace('{{NBA}}',''.join(nrows)).replace('{{ALL}}',''.join(allh)))
open('/home/user/dimsas12websites2/betting/sharpe_analysis.html','w').write(out); print(len(out))
