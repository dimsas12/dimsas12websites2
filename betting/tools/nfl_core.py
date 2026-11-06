import json,urllib.request,gzip,sys,os
from concurrent.futures import ThreadPoolExecutor
B=sys.argv[1]; EV=sys.argv[2]; UA={"User-Agent":"Mozilla/5.0","Accept-Encoding":"gzip"}
def get(u):
    b=urllib.request.urlopen(urllib.request.Request(u.replace('http://','https://'),headers=UA),timeout=40).read()
    if b[:2]==b'\x1f\x8b': b=gzip.decompress(b)
    return json.loads(b)
items=[];p=1
while True:
    d=get(f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{EV}/competitions/{EV}/odds/100/propBets?lang=en&region=us&limit=500&page={p}")
    items+=d.get('items',[])
    if p>=d.get('pageCount',1): break
    p+=1
refs=sorted({x['athlete']['$ref'] for x in items if x.get('athlete')})
A={}
def f(r):
    try: d=get(r); return r,d.get('displayName')
    except Exception: return r,None
with ThreadPoolExecutor(16) as ex:
    for r,v in ex.map(f,refs): A[r]=v
out=[dict(type=x['type']['name'],player=A.get((x.get('athlete') or {}).get('$ref')),aid=((x.get('athlete') or {}).get('$ref') or '').split('athletes/')[-1].split('?')[0],raw=x.get('current'),odds=x.get('odds'),upd=x.get('lastUpdated')) for x in items]
json.dump(out,open(B+'/props.json','w'))
from collections import Counter
print(len(out),'props',len(refs),'athletes'); print(Counter(o['type'] for o in out).most_common(40))
sm=get(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={EV}"); json.dump(sm,open(B+'/game.json','w'))
print([(p.get('provider',{}).get('name'),p.get('details'),p.get('overUnder')) for p in sm.get('pickcenter',[])])
for t in sm.get('injuries',[]): print(t['team']['abbreviation'],[(i['athlete']['displayName'],i['athlete'].get('position',{}).get('abbreviation'),i['status']) for i in t['injuries']])
