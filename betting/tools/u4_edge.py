import json,glob,sys,re,math
from collections import defaultdict
S=sys.argv[1]; P=json.load(open(S+'/u4/props.json')); A=json.load(open(S+'/u4/ath.json'))
# athlete id from ref
ref2id={r:re.search(r'athletes/(\d+)',r).group(1) for r in A}
logs=defaultdict(list); names={}
for f in glob.glob(S+'/u4/sum/*.json'):
    d=json.load(open(f)); c=d['header']['competitions'][0]
    for side in d.get('rosters',[]):
        for x in side.get('roster',[]):
            st={s['name']:s.get('value',0) or 0 for s in x.get('stats',[])}
            if not st.get('appearances'): continue
            aid=x['athlete']['id']; names[aid]=x['athlete']['displayName']
            logs[aid].append(dict(date=c['date'],team=side['team']['displayName'],start=x.get('starter'),subout=x.get('subbedOut'),pos=(x.get('position') or {}).get('abbreviation'),
                sh=st.get('totalShots',0),sot=st.get('shotsOnTarget',0),fc=st.get('foulsCommitted',0),fs=st.get('foulsSuffered',0),ast=st.get('goalAssists',0),off=st.get('offsides',0),yc=st.get('yellowCards',0),sv=st.get('saves',0) or max(0,st.get('shotsFaced',0)-st.get('goalsConceded',0)),g=st.get('totalGoals',0)))
for v in logs.values(): v.sort(key=lambda g:g['date'])
M={'Shots Milestones':'sh','Shots on Target Milestones':'sot','Fouls Committed Milestones':'fc','Fouls Milestones':'fs','Assists Milestones':'ast','Offsides Milestones':'off','Goalkeeper Saves Milestones':'sv'}
G={}
for f in glob.glob(S+'/u4/4019*.json'):
    d=json.load(open(f)); c=d['header']['competitions'][0]; G[f.split('/')[-1][:-5]]=' v '.join(x['team']['displayName'] for x in sorted(c['competitors'],key=lambda x:x['homeAway'],reverse=True))
rows=[]
# need athlete ref per prop: re-read raw items? props.json has player name only; map name->id via names
n2id=defaultdict(list)
for k,v in names.items(): n2id[v].append(k)
for p in P:
    k=M.get(p['type'])
    if not k or not p['player'] or not p['raw'].get('target'): continue
    tgt=p['raw']['target']['value']; od=p['raw'].get('over',{})
    if not od: continue
    dec=od.get('decimal'); am=od.get('american')
    ids=n2id.get(p['player'],[])
    if not ids: continue
    L=logs[ids[0]]
    st=[g for g in L if g['start']]
    if len(st)<5: continue
    last=st[-10:]
    hits=sum(g[k]>=tgt for g in last); hr=hits/len(last)
    allh=sum(g[k]>=tgt for g in st)/len(st)
    # Poisson on per-start mean, shrink 50% toward market implied
    mu=sum(g[k] for g in last)/len(last)
    pp=1-sum(math.exp(-mu)*mu**i/math.factorial(i) for i in range(int(tgt)))
    mkt=1/dec
    est=(pp+allh+mkt)/3
    rows.append(dict(game=G[p['gid']],player=p['player'],mk=p['type'].replace(' Milestones',''),tgt=tgt,am=am,dec=dec,hits=f"{hits}/{len(last)}",hr=hr,allstarts=len(st),allh=round(allh,2),mu=round(mu,2),pois=round(pp,2),est=round(est,3),ev=round(est*dec-1,3),vals=[int(g[k]) for g in last],recent_start=L[-1]['start'] if L else None,last3=[('S' if g['start'] else 'b') for g in L[-3:]],upd=p['upd']))
json.dump(rows,open(S+'/u4/edges.json','w'))
ok=[r for r in rows if 1.65<=r['dec']<=2.51]
print(len(rows),'evaluable lines;',len(ok),'in odds range')
for r in sorted(ok,key=lambda r:-r['ev'])[:45]:
    print(f"{r['game'][:34]:34s} {r['player'][:20]:20s} {r['mk'][:16]:16s} {int(r['tgt'])}+ {r['am']:>5s} ({r['dec']:.2f}) L10 {r['hits']:5s} all {r['allh']:.2f}/{r['allstarts']} mu {r['mu']:.2f} pois {r['pois']:.2f} est {r['est']:.2f} EV {r['ev']:+.2f} {r['vals']} {''.join(r['last3'])} upd {r['upd'][:10]}")
