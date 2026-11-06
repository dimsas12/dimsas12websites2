import json,urllib.request,gzip,sys,os,glob
from concurrent.futures import ThreadPoolExecutor
S=sys.argv[1]; UA={"User-Agent":"Mozilla/5.0","Accept-Encoding":"gzip"}
def get(u):
    b=urllib.request.urlopen(urllib.request.Request(u.replace('http://','https://'),headers=UA),timeout=40).read()
    if b[:2]==b'\x1f\x8b': b=gzip.decompress(b)
    return json.loads(b)
ids=[os.path.basename(f)[:-5] for f in glob.glob(S+'/u4/4019*.json')]
items=[]
for i in ids:
    p=1
    while True:
        d=get(f"https://sports.core.api.espn.com/v2/sports/soccer/leagues/uefa.champions/events/{i}/competitions/{i}/odds/100/propBets?lang=en&region=us&limit=500&page={p}")
        for x in d['items']: x['gid']=i
        items+=d['items']
        if p>=d.get('pageCount',1): break
        p+=1
refs=sorted({x['athlete']['$ref'] for x in items if x.get('athlete')})
cache=S+'/u4/ath.json'; A=json.load(open(cache)) if os.path.exists(cache) else {}
def f(r):
    try: d=get(r); return r,(d.get('displayName'),d.get('position',{}).get('abbreviation'),(d.get('team') or {}).get('$ref',''))
    except Exception as e: return r,(None,None,'')
with ThreadPoolExecutor(16) as ex:
    for r,v in ex.map(f,[r for r in refs if r not in A]): A[r]=v
json.dump(A,open(cache,'w'))
out=[]
for x in items:
    a=A.get(x.get('athlete',{}).get('$ref'),(None,None,''))
    c=x.get('current',{})
    out.append(dict(gid=x['gid'],type=x['type']['name'],player=a[0],pos=a[1],target=c.get('target',{}).get('value') if isinstance(c.get('target'),dict) else None,
                    over=(c.get('over') or {}).get('american'),under=(c.get('under') or {}).get('american'),raw=c,upd=x.get('lastUpdated')))
json.dump(out,open(S+'/u4/props.json','w'),default=str)
from collections import Counter
print(len(items),'props;',len(refs),'athletes'); print(Counter(o['type'] for o in out).most_common(40))
print(json.dumps(out[0]['raw'])[:400]); 
ex=[o for o in out if 'Shots' in o['type']][:3]; print([ (o['player'],o['type'],json.dumps(o['raw'])[:300]) for o in ex])
