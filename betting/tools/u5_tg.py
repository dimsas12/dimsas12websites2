import json,urllib.request,gzip,sys,os
from concurrent.futures import ThreadPoolExecutor
S=sys.argv[1]; UA={"User-Agent":"Mozilla/5.0","Accept-Encoding":"gzip"}
def get(u):
    b=urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=40).read()
    if b[:2]==b'\x1f\x8b': b=gzip.decompress(b)
    return json.loads(b)
F=json.load(open(S+'/u5/form.json')); sel={}
for tid in F['T']:
    rs=sorted({(r[1],r[2],r[7]) for r in F['R'] if r[0]==tid})[-12:]
    for dt,lg,i in rs: sel[i]=lg
def fetch(it):
    i,lg=it; p=f"{S}/u5/sum/{i}.json"
    if os.path.exists(p): return 0
    try: json.dump(get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{lg}/summary?event={i}"),open(p,'w')); return 1
    except Exception: return 0
with ThreadPoolExecutor(24) as ex: n=sum(ex.map(fetch,sel.items()))
print(len(sel),'games',n,'downloaded')
