import json,urllib.request,gzip,sys,os
from concurrent.futures import ThreadPoolExecutor
S=sys.argv[1]; D=sys.argv[2]; UA={"User-Agent":"Mozilla/5.0","Accept-Encoding":"gzip"}
def get(u):
    b=urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=40).read()
    if b[:2]==b'\x1f\x8b': b=gzip.decompress(b)
    return json.loads(b)
LG=['uefa.champions','uefa.europa','uefa.europa.conf','eng.1','eng.2','esp.1','ita.1','ger.1','fra.1','tur.1','por.1','ned.1','aut.1','nor.1','swe.1','den.1','sui.1','bel.1','gre.1','cze.1','pol.1','cro.1','srb.1','bul.1','rou.1','sco.1','cyp.1','isr.1','hun.1','fin.1','alb.1','bih.1','ltu.1','lva.1','gib.1','arm.1','and.1','svn.1','kaz.1']
G=[]
for lg in ['eng.1','esp.1','ita.1','ger.1','fra.1','tur.1','por.1','ned.1']:
    for e in get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{lg}/scoreboard?dates={D}")['events']: G.append((lg,e['id']))
def summ(x):
    lg,i=x; d=get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{lg}/summary?event={i}"); d['_lg']=lg
    json.dump(d,open(f"{S}/u6/g{i}.json",'w')); return d
with ThreadPoolExecutor(10) as ex: SU=list(ex.map(summ,G))
T={}
for d in SU:
    for c in d['header']['competitions'][0]['competitors']: T[c['team']['id']]=c['team']['displayName']
def sched(x):
    tid,lg=x
    try: s=get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{lg}/teams/{tid}/schedule?season=2026")
    except Exception: return []
    out=[]
    for g in s.get('events',[]):
        cc=g['competitions'][0]
        if not cc['status']['type'].get('completed'): continue
        try: sc={x['team']['id']:int(float(x['score']['value'])) for x in cc['competitors']}
        except Exception: continue
        if tid not in sc: continue
        opp=[x['team']['displayName'] for x in cc['competitors'] if x['team']['id']!=tid][0]
        h=[x['homeAway'] for x in cc['competitors'] if x['team']['id']==tid][0]
        out.append((tid,g['date'][:10],lg,h[0],opp[:14],sc[tid],[v for k,v in sc.items() if k!=tid][0],g['id']))
    return out
jobs=[(t,l) for t in T for l in LG]
with ThreadPoolExecutor(24) as ex: R=[r for rs in ex.map(sched,jobs) for r in rs]
json.dump({'T':T,'R':R},open(S+'/u6/form.json','w'))
for d in SU:
    c=d['header']['competitions'][0]; o=(d.get('pickcenter') or [{}])[0]
    nm=' v '.join(x['team']['displayName'] for x in sorted(c['competitors'],key=lambda x:x['homeAway'],reverse=True))
    print('\n###',c['date'],d['_lg'],nm,'|',o.get('provider',{}).get('name'),'O/U',o.get('overUnder'),o.get('overOdds'),o.get('underOdds'),'ML',o.get('homeTeamOdds',{}).get('moneyLine'),o.get('drawOdds',{}).get('moneyLine'),o.get('awayTeamOdds',{}).get('moneyLine'))
    for x in sorted(c['competitors'],key=lambda x:x['homeAway'],reverse=True):
        tid=x['team']['id']; rs=sorted(set(r[1:] for r in R if r[0]==tid))[-10:]
        gf=sum(r[4] for r in rs)/max(1,len(rs)); ga=sum(r[5] for r in rs)/max(1,len(rs))
        print(f"   {x['homeAway']:4s} {x['team']['displayName']:22s} last{len(rs)} GF {gf:.2f} GA {ga:.2f} |", '; '.join(f"{r[0][5:]} {r[1][:4]} {r[2]} {r[3]} {r[4]}-{r[5]}" for r in rs))
