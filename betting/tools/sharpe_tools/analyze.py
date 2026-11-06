import json,sys,re,unicodedata,statistics as st
from datetime import datetime,timedelta
S=sys.argv[1]; B=json.load(open(S+'/shp/box.json')); P=B['players']; G=B['games']
R=[r for r in json.load(open(S+'/sharpe.json'))['rows'] if r['sport']=='NBA']
def norm(s): return re.sub(r'[^a-z ]','',unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode().lower()).replace(' iii','').replace(' jr','').strip()
idx={norm(v['name']):k for k,v in P.items()}; alias={'robert willams':'robert williams'}
F={'points':lambda g:g['pts'],'rebounds':lambda g:g['reb'],'assists':lambda g:g['ast'],'3pm':lambda g:g['tpm'],'pra':lambda g:g['pts']+g['reb']+g['ast'],'pr':lambda g:g['pts']+g['reb'],'pa':lambda g:g['pts']+g['ast'],'ra':lambda g:g['reb']+g['ast']}
def lastseason(aid):
    try: d=json.load(open(f"{S}/shp/gl/{aid}.json"))
    except: return []
    if 'seasonTypes' not in d: return []
    lab=d['labels']; out=[]
    for stt in d['seasonTypes']:
        if 'Regular' not in stt['displayName']: continue
        for c in stt['categories']:
            for e in c.get('events',[]):
                v=dict(zip(lab,e['stats'])); dt=d['events'].get(e['eventId'],{}).get('gameDate','')
                out.append(dict(date=dt,min=float(v['MIN']),pts=float(v['PTS']),reb=float(v['REB']),ast=float(v['AST']),tpm=float(v['3PT'].split('-')[0]),tpa=float(v['3PT'].split('-')[1])))
    return sorted(out,key=lambda g:g['date'])
res=[]
for r in R:
    m=re.match(r'([ou])([\d.]+)\s+(.+)',r['prop'],re.I)
    n=alias.get(norm(r['who']),norm(r['who'])); aid=idx.get(n)
    if not aid: continue
    if not m: res.append(dict(r=r,note='non-line prop')); continue
    side,line,mk=m.group(1).lower(),float(m.group(2)),m.group(3).lower().strip()
    f=F.get(mk)
    if not f: res.append(dict(r=r,note='market '+mk)); continue
    d0=datetime.fromisoformat(r['date']); lo=(d0+timedelta(hours=14)).isoformat(); hi=(d0+timedelta(hours=34)).isoformat()
    gl=P[aid]['g']; game=[g for g in gl if lo<=g['date'][:19]<=hi]
    prior=[g for g in gl if g['date'][:19]<lo]
    ls=lastseason(aid)
    def hr(gs): 
        v=[f(g) for g in gs]; 
        if not v: return None
        return dict(n=len(v),avg=round(st.mean(v),1),hit=sum((x>line) if side=='o' else (x<line) for x in v),vals=v,min=round(st.mean(g['min'] for g in gs),1))
    info=dict(r=r,side=side,line=line,mk=mk,aid=aid,name=P[aid]['name'],ls=hr(ls),ls15=hr(ls[-15:]),cur=hr(prior))
    if game:
        g=game[0]; info['actual']=f(g); info['g']=g
        team=g['team']; gid=g['gid']
        # teammates with >=24 min avg (prior this season, else last season unknown) missing tonight
        mates={}
        for k,v in P.items():
            pg=[x for x in v['g'] if x['team']==team and x['date'][:19]<lo]
            if pg and st.mean(x['min'] for x in pg)>=24 and k!=aid: mates[k]=v['name']
        played={k for k,v in P.items() for x in v['g'] if x['gid']==gid and x['team']==team}
        info['out_mates']=[nm for k,nm in mates.items() if k not in played]
        info['dnp']=G[gid][team]['dnp']
        info['opp']=g['opp']; info['started']=g['starter']; info['gmin']=g['min']
        # prior game role: did he start / minutes trend
        if prior: info['last']=[(round(x['min']),f(x)) for x in prior[-3:]]
    res.append(info)
json.dump(res,open(S+'/shp/nba_analysis.json','w'),default=str)
for i in res:
    r=i['r']
    if 'note' in i: print(r['date'],r['who'],r['prop'],r['res'],'|',i['note']); continue
    def s(h): return f"{h['hit']}/{h['n']} avg {h['avg']} ({h['min']}m)" if h else '—'
    print(f"{r['date']} {r['res']} {i['name']:24s} {i['side']}{i['line']} {i['mk']:8s} {r['odds']:+.0f} | LS {s(i['ls'])} L15 {s(i['ls15'])} | CUR {s(i['cur'])} {i.get('last','')} | act {i.get('actual')} {i.get('gmin')}m vs {i.get('opp')} out:{i.get('out_mates')}")
