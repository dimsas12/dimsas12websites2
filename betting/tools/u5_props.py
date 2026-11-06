import json,urllib.request,gzip,sys,os,glob
from concurrent.futures import ThreadPoolExecutor
S=sys.argv[1]; UA={"User-Agent":"Mozilla/5.0","Accept-Encoding":"gzip"}
def get(u):
    b=urllib.request.urlopen(urllib.request.Request(u.replace('http://','https://'),headers=UA),timeout=40).read()
    if b[:2]==b'\x1f\x8b': b=gzip.decompress(b)
    return json.loads(b)
G=[]
for f in glob.glob(S+'/u5/g*.json'):
    d=json.load(open(f)); G.append((d['_lg'],os.path.basename(f)[1:-5]))
def pull(x):
    lg,i=x; items=[]; p=1
    while True:
        try: d=get(f"https://sports.core.api.espn.com/v2/sports/soccer/leagues/{lg}/events/{i}/competitions/{i}/odds/100/propBets?lang=en&region=us&limit=500&page={p}")
        except Exception: break
        for it in d.get('items',[]): it['gid']=i
        items+=d.get('items',[])
        if p>=d.get('pageCount',1): break
        p+=1
    return items
with ThreadPoolExecutor(10) as ex: items=[i for r in ex.map(pull,G) for i in r]
refs=sorted({x['athlete']['$ref'] for x in items if x.get('athlete')})
A={}
def f(r):
    try: d=get(r); return r,(d.get('displayName'),None,'')
    except Exception: return r,(None,None,'')
with ThreadPoolExecutor(20) as ex:
    for r,v in ex.map(f,refs): A[r]=v
json.dump(A,open(S+'/u5/ath.json','w'))
out=[]
for x in items:
    a=A.get((x.get('athlete') or {}).get('$ref'),(None,None,''))
    c=x.get('current',{})
    out.append(dict(gid=x['gid'],type=x['type']['name'],player=a[0],raw=c,odds=x.get('odds'),upd=x.get('lastUpdated')))
json.dump(out,open(S+'/u5/props.json','w'),default=str)
from collections import Counter
print(len(items),'props;',len(refs),'athletes', Counter(o['gid'] for o in out).most_common(40))
