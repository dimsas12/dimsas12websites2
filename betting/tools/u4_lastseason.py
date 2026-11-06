import json,urllib.request,gzip,sys,os
from concurrent.futures import ThreadPoolExecutor
S=sys.argv[1]; tid=sys.argv[2]; leagues=sys.argv[3].split(','); who=sys.argv[4]; stat=sys.argv[5]
UA={"User-Agent":"Mozilla/5.0","Accept-Encoding":"gzip"}
def get(u):
    b=urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=40).read()
    if b[:2]==b'\x1f\x8b': b=gzip.decompress(b)
    return json.loads(b)
ev=[]
for lg in leagues:
    try: s=get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{lg}/teams/{tid}/schedule?season=2025")
    except Exception: continue
    ev+=[(g['date'],lg,g['id']) for g in s.get('events',[]) if g['competitions'][0]['status']['type'].get('completed')]
def f(x):
    dt,lg,i=x; p=f"{S}/u4/ls/{i}.json"
    if not os.path.exists(p):
        try: json.dump(get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{lg}/summary?event={i}"),open(p,'w'))
        except Exception: return None
    d=json.load(open(p))
    for side in d.get('rosters',[]):
        for a in side.get('roster',[]):
            if who.lower() in a['athlete']['displayName'].lower():
                st={s['name']:s.get('value',0) or 0 for s in a.get('stats',[])}
                if not st.get('appearances'): return None
                opp=[t['team']['displayName'] for t in d['header']['competitions'][0]['competitors'] if t['team']['id']!=tid]
                ha=[t['homeAway'] for t in d['header']['competitions'][0]['competitors'] if t['team']['id']==tid]
                v=st.get(stat,0)
                if stat=='saves' and not v: v=max(0,st.get('shotsFaced',0)-st.get('goalsConceded',0))
                return (dt[:10],lg[:4],ha[0][0] if ha else '?',opp[0][:14] if opp else '',a.get('starter'),v)
    return None
with ThreadPoolExecutor(12) as ex: R=[r for r in ex.map(f,sorted(ev)) if r]
st=[r for r in R if r[4]]
print(who,'last season starts',len(st))
for r in st: print('  ',r)
for t in (2,3,4,5): print(f"  >= {t}: {sum(r[5]>=t for r in st)}/{len(st)} = {sum(r[5]>=t for r in st)/max(1,len(st)):.2f}")
print('  mean',sum(r[5] for r in st)/max(1,len(st)))
