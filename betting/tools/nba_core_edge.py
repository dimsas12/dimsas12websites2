"""Evaluate DraftKings NBA milestone props (ESPN core) vs current + last-season logs."""
import json,sys,os,re,math,unicodedata,urllib.request,gzip,statistics as stt
from concurrent.futures import ThreadPoolExecutor
base=sys.argv[1]; P=json.load(open(base+'/core_props.json')); A=json.load(open(base+'/ath.json')); PL=json.load(open(base+'/players.json'))
UA={"User-Agent":"Mozilla/5.0","Accept-Encoding":"gzip"}
def get(u):
    b=urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=40).read()
    if b[:2]==b'\x1f\x8b': b=gzip.decompress(b)
    return json.loads(b)
def norm(s): return re.sub(r'[^a-z ]','',unicodedata.normalize('NFKD',s or '').encode('ascii','ignore').decode().lower()).replace(' iii','').replace(' jr','').replace(' ii','').strip()
cur={norm(k.split('|')[1]):(k.split('|')[0],v) for k,v in PL.items()}
M={'Points Milestones':'pts','Rebounds Milestones':'reb','Assists Milestones':'ast','3-Point Field Goals Milestones':'tpm','Points + Assists Milestones':'pa','Points + Rebounds Milestones':'pr','Rebounds + Assists Milestones':'ra','Points + Assists + Rebounds Milestones':'pra'}
F={'pts':lambda g:g['pts'],'reb':lambda g:g['reb'],'ast':lambda g:g['ast'],'tpm':lambda g:g['tpm'],'pa':lambda g:g['pts']+g['ast'],'pr':lambda g:g['pts']+g['reb'],'ra':lambda g:g['reb']+g['ast'],'pra':lambda g:g['pts']+g['reb']+g['ast']}
name_of={}; aid_of={}
for r,n in A.items():
    m=re.search(r'athletes/(\d+)',r)
    if m and n: aid_of[n]=m.group(1)
os.makedirs(base+'/gl',exist_ok=True)
def last(aid):
    p=f"{base}/gl/{aid}.json"
    if not os.path.exists(p):
        try: json.dump(get(f"https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{aid}/gamelog?season=2026"),open(p,'w'))
        except Exception as e: json.dump({'err':str(e)},open(p,'w'))
    d=json.load(open(p)); out=[]
    if 'names' not in d: return out
    nm=d['names']
    for stp in d.get('seasonTypes',[]):
        if 'Regular' not in stp['displayName']: continue
        for c in stp['categories']:
            for e in c.get('events',[]):
                r=dict(zip(nm,e['stats']))
                try: mn=float(r['minutes'])
                except: continue
                if mn<=0: continue
                out.append(dict(min=mn,pts=int(r['points']),reb=int(r['totalRebounds']),ast=int(r['assists']),tpm=int(r['threePointFieldGoalsMade-threePointFieldGoalsAttempted'].split('-')[0])))
    return out
with ThreadPoolExecutor(12) as ex: LS=dict(zip(aid_of.keys(),ex.map(last,aid_of.values())))
from collections import defaultdict
grp=defaultdict(list)
for p in P:
    if p['type'] not in M or not p['player'] or not p.get('odds'): continue
    try: float(p['odds']['decimal']['value'])
    except Exception: continue
    grp[(p['player'],p['type'])].append(p)
keep=[]
for k,v in grp.items():
    lad=[[]]
    for p in v:
        if lad[-1] and p['raw']['target']['value']<=lad[-1][-1]['raw']['target']['value']: lad.append([])
        lad[-1].append(p)
    def score(L): return max(x['raw']['target']['value'] for x in L)
    keep+=max(lad,key=score)
P=keep
rows=[]
for p in P:
    k=M.get(p['type'])
    if not k or not p['player'] or not p.get('odds'): continue
    try: dec=float(p['odds']['decimal']['value']); am=p['odds']['american']['value']; op=p['odds']['american'].get('open')
    except Exception: continue
    t=p['raw']['target']['value']
    c=cur.get(norm(p['player'])); cg=c[1] if c else []
    lg=LS.get(p['player'],[])
    cs=[F[k](g) for g in cg]; ls=[F[k](g) for g in lg]
    if len(cs)+len(ls)<10: continue
    hc=sum(x>=t for x in cs); hl=sum(x>=t for x in ls); n=len(cs)+len(ls)
    big=(hc+hl)/n
    # weighted mean: current x2
    mu_c=stt.mean(cs) if cs else None; mu_l=stt.mean(ls) if ls else None
    mu=(2*mu_c*len(cs)+mu_l*len(ls))/(2*len(cs)+len(ls)) if cs and ls else (mu_c if cs else mu_l)
    ln=t-0.5
    sd={'pts':0.28*mu+2.5,'reb':0.30*mu+1.3,'ast':0.32*mu+1.0,'pra':0.22*mu+3.5,'pr':0.24*mu+3.0,'pa':0.24*mu+3.0,'ra':0.28*mu+1.5,'tpm':None}[k]
    pm=1-sum(math.exp(-mu)*mu**i/math.factorial(i) for i in range(int(t))) if k=='tpm' else 0.5*math.erfc((ln-mu)/sd/math.sqrt(2))
    est=(pm+big+1/dec)/3
    rows.append(dict(game=p['game'],player=p['player'],team=c[0] if c else '?',mk=k,tgt=t,am=am,open=op,dec=dec,ncur=len(cs),nls=len(ls),hc=f"{hc}/{len(cs)}",hl=f"{hl}/{len(ls)}",big=round(big,2),mu=round(mu,1),margin=round((mu-ln)/ln,2),pm=round(pm,2),est=round(est,3),ev=round(est*dec-1,3),last=cs[-6:],mins=[round(g['min']) for g in cg[-6:]]))
json.dump(rows,open(base+'/core_edges.json','w'))
ok=[r for r in rows if 1.65<=r['dec']<=2.51]
print(len(rows),'lines evaluated;',len(ok),'in range;',len({r['player'] for r in rows}),'players')
for r in sorted(ok,key=lambda r:-r['ev'])[:60]:
    print(f"{r['game']:10s} {r['player'][:20]:20s} {r['mk']:4s} {int(r['tgt'])}+ {r['am']:>5s}(op {r['open']:>5s}) {r['dec']:.2f} | cur {r['hc']:5s} ls {r['hl']:6s} big {r['big']:.2f} mu {r['mu']:5} marg {r['margin']:+.2f} | est {r['est']:.2f} EV {r['ev']:+.2f} {r['last']} min {r['mins']}")
