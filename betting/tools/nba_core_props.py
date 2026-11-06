"""DraftKings NBA props from ESPN core odds API (provider 100 = DraftKings)."""
import json,urllib.request,gzip,sys,os
from concurrent.futures import ThreadPoolExecutor
base=sys.argv[1]; UA={"User-Agent":"Mozilla/5.0","Accept-Encoding":"gzip"}
def get(u):
    b=urllib.request.urlopen(urllib.request.Request(u.replace('http://','https://'),headers=UA),timeout=40).read()
    if b[:2]==b'\x1f\x8b': b=gzip.decompress(b)
    return json.loads(b)
sb=json.load(open(base+'/sb.json')); items=[]
for e in sb['events']:
    i=e['id']; p=1
    while True:
        try: d=get(f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{i}/competitions/{i}/odds/100/propBets?lang=en&region=us&limit=500&page={p}")
        except Exception as ex: print(i,'ERR',ex); break
        for x in d.get('items',[]): x['gid']=i; x['game']=e['shortName']
        items+=d.get('items',[])
        if p>=d.get('pageCount',1): break
        p+=1
refs=sorted({x['athlete']['$ref'] for x in items if x.get('athlete')})
cache=base+'/ath.json'; A=json.load(open(cache)) if os.path.exists(cache) else {}
def f(r):
    try: d=get(r); return r,d.get('displayName')
    except Exception: return r,None
with ThreadPoolExecutor(16) as ex:
    for r,v in ex.map(f,[r for r in refs if r not in A]): A[r]=v
json.dump(A,open(cache,'w'))
out=[]
for x in items:
    c=x.get('current',{})
    out.append(dict(gid=x['gid'],game=x['game'],type=x['type']['name'],player=A.get((x.get('athlete') or {}).get('$ref')),raw=c,odds=x.get('odds'),upd=x.get('lastUpdated')))
json.dump(out,open(base+'/core_props.json','w'))
from collections import Counter
print(len(out),'props',len(refs),'athletes'); print(Counter(o['type'] for o in out).most_common(30))
for o in out[:3]: print(o['player'],o['type'],json.dumps(o['raw'])[:300],o['upd'])
