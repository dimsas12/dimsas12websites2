import json,urllib.request,gzip,sys,os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
B=sys.argv[1]; UA={"User-Agent":"Mozilla/5.0","Accept-Encoding":"gzip"}
def get(u):
    b=urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=40).read()
    if b[:2]==b'\x1f\x8b': b=gzip.decompress(b)
    return json.loads(b)
g=json.load(open(B+'/game.json')); teams={c['team']['abbreviation']:c['team']['id'] for c in g['header']['competitions'][0]['competitors']}
ids=[]
for ab,tid in teams.items():
    s=get(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{tid}/schedule?season=2026&seasontype=2")
    for e in s.get('events',[]):
        if e['competitions'][0]['status']['type'].get('completed'): ids.append(e['id'])
def f(i):
    p=f"{B}/sum/{i}.json"
    if not os.path.exists(p): json.dump(get(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={i}"),open(p,'w'))
    return json.load(open(p))
with ThreadPoolExecutor(10) as ex: S=list(ex.map(f,sorted(set(ids))))
L=defaultdict(list); res=defaultdict(list)
for d in S:
    c=d['header']['competitions'][0]; dt=c['date'][:10]
    sc={x['team']['abbreviation']:int(x.get('score') or 0) for x in c['competitors']}
    for ab in sc:
        if ab in teams: o=[k for k in sc if k!=ab][0]; res[ab].append((dt,o,sc[ab],sc[o]))
    for t in d.get('boxscore',{}).get('players',[]):
        ab=t['team']['abbreviation']
        if ab not in teams: continue
        row=defaultdict(dict)
        for cat in t['statistics']:
            for a in cat['athletes']:
                row[(a['athlete']['id'],a['athlete']['displayName'])].update({cat['name']+'.'+k:v for k,v in zip(cat['keys'],a['stats'])})
        for (aid,nm),r in row.items(): L[aid].append(dict(date=dt,team=ab,name=nm,**r))
json.dump({k:sorted(v,key=lambda x:x['date']) for k,v in L.items()},open(B+'/logs.json','w'))
for ab in teams: print(ab,sorted(res[ab]))
