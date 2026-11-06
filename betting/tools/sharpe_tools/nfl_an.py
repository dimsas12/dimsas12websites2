import json,sys,re,unicodedata,os,urllib.request,gzip,statistics as st
from datetime import datetime,timedelta
S=sys.argv[1]; B=json.load(open(S+'/shp/nfl_box.json')); P=B['players']; G=B['games']
R=[r for r in json.load(open(S+'/sharpe.json'))['rows'] if r['sport']=='NFL']
def norm(s): return re.sub(r'[^a-z ]','',unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode().lower()).replace(' iii','').replace(' ii','').replace(' jr','').strip()
idx={}
for k,v in P.items(): idx.setdefault(norm(v['name']),k)
UA={"User-Agent":"Mozilla/5.0","Accept-Encoding":"gzip"}
def get(u):
    b=urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=40).read()
    if b[:2]==b'\x1f\x8b': b=gzip.decompress(b)
    return json.loads(b)
def lastseason(aid):
    p=f"{S}/shp/nfl/gl_{aid}.json"
    if not os.path.exists(p):
        try: json.dump(get(f"https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{aid}/gamelog?season=2025"),open(p,'w'))
        except Exception as e: json.dump({'err':str(e)},open(p,'w'))
    d=json.load(open(p)); out=[]
    if 'seasonTypes' not in d: return out
    names=d['names']
    for stt in d['seasonTypes']:
        if 'Regular' not in stt['displayName']: continue
        for c in stt['categories']:
            for e in c.get('events',[]):
                v={}
                for n_,x in zip(names,e['stats']):
                    if n_ in v: n_='r_'+n_
                    v[n_]=x
                def f(k):
                    try: return float(str(v.get(k,'0')).replace(',',''))
                    except: return 0.0
                out.append(dict(rec=f('receptions'),yds=f('receivingYards'),tgt=f('receivingTargets'),long=f('longReception'),ratt=f('rushingAttempts'),ryds=f('r_rushingYards') if 'r_rushingYards' in v else f('rushingYards'),rlong=f('longRushing'),pyds=f('passingYards'),patt=f('passingAttempts'),cmp=f('completions'),rtd=f('rushingTouchdowns'),ctd=f('receivingTouchdowns')))
    return out
F={'receiving yards':lambda g:g.get('yds',0),'receptions':lambda g:g.get('rec',0),'rushing yards':lambda g:g.get('ryds',0),'rush attempts':lambda g:g.get('ratt',0),'rushing attempts':lambda g:g.get('ratt',0),
   'longest reception':lambda g:g.get('long',0),'longest rush':lambda g:g.get('rlong',0),'passing yards':lambda g:g.get('pyds',0),'pass attempts':lambda g:g.get('patt',0),'pass completions':lambda g:g.get('cmp',0),'completions':lambda g:g.get('cmp',0),
   'pass + rush yards':lambda g:g.get('pyds',0)+g.get('ryds',0),'rr':lambda g:g.get('yds',0)+g.get('ryds',0)}
res=[]
for r in R:
    m=re.match(r'([ou])([\d.]+)\s+(.+)',r['prop'].strip(),re.I)
    aid=idx.get(norm(r['who']))
    if not aid: res.append(dict(r=r,note='player not found in 2026 NFL box scores')); continue
    d0=datetime.fromisoformat(r['date']); lo=(d0-timedelta(hours=12)).isoformat(); hi=(d0+timedelta(hours=40)).isoformat()
    gl=P[aid]['g']; game=[g for g in gl if lo<=g['date'][:19]<=hi]; prior=[g for g in gl if g['date'][:19]<lo]
    info=dict(r=r,aid=aid,name=P[aid]['name'],pos=P[aid]['pos'])
    if not m:
        info['note']='non-line prop ('+r['prop']+')'
        if game: info['actual_td']=game[0].get('rtd',0)+game[0].get('ctd',0)
        res.append(info); continue
    side,line,mk=m.group(1).lower(),float(m.group(2)),m.group(3).lower().strip()
    f=F.get(mk)
    if not f: info['note']='market '+mk; res.append(info); continue
    ls=lastseason(aid)
    def hr(gs):
        v=[f(g) for g in gs]
        if not v: return None
        return dict(n=len(v),avg=round(st.mean(v),1),hit=sum((x>line) if side=='o' else (x<line) for x in v),vals=v,
                    tgt=round(st.mean(g.get('tgt',0) for g in gs),1),car=round(st.mean(g.get('ratt',0) for g in gs),1))
    info.update(side=side,line=line,mk=mk,ls=hr(ls),cur=hr(prior))
    if game:
        g=game[0]; info['actual']=f(g); info['g']=g; gm=G[g['gid']]
        sp=gm['spread_home']; info['spread']=None if sp is None else (sp if g['home'] else -sp); info['total']=gm['total']; info['opp']=g['opp']
        # teammates top-target earners missing
        mates=[]
        for k,v in P.items():
            pg=[x for x in v['g'] if x['team']==g['team'] and x['date'][:19]<lo]
            if k!=aid and pg and st.mean(x.get('tgt',0) for x in pg)>=5: mates.append(k)
        played={k for k,v in P.items() for x in v['g'] if x['gid']==g['gid'] and x['team']==g['team']}
        info['out_mates']=[P[k]['name'] for k in mates if k not in played]
        info['tgt_game']=g.get('tgt'); info['car_game']=g.get('ratt')
    res.append(info)
json.dump(res,open(S+'/shp/nfl_analysis.json','w'),default=str)
for i in res:
    r=i['r']
    if 'note' in i: print(r['date'],r['res'],r['who'],'|',r['prop'],'|',i['note'],i.get('actual_td','')); continue
    def s(h): return f"{h['hit']}/{h['n']} avg {h['avg']} (tgt {h['tgt']} car {h['car']})" if h else '—'
    print(f"{r['date']} {r['res']} {i['name']:22s} {i['pos']} {i['side']}{i['line']} {i['mk']:16s} {r['odds']:+.0f} | LS {s(i['ls'])} | CUR {s(i['cur'])} {i['cur']['vals'] if i['cur'] else ''} | act {i.get('actual')} tgt {i.get('tgt_game')} car {i.get('car_game')} sp {i.get('spread')} tot {i.get('total')} out {i.get('out_mates')}")
