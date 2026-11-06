import json,urllib.request,gzip,sys,glob,os
from concurrent.futures import ThreadPoolExecutor
S=sys.argv[1]; UA={"User-Agent":"Mozilla/5.0","Accept-Encoding":"gzip"}
def get(u):
    b=urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=40).read()
    if b[:2]==b'\x1f\x8b': b=gzip.decompress(b)
    return json.loads(b)
LG=['uefa.champions','eng.1','esp.1','ita.1','ger.1','fra.1','tur.1','por.1','ned.1','aut.1','nor.1','ukr.1','svk.1','gre.1','cze.1','bel.1','aze.1','uefa.europa','gre.cup','fra.coupe_de_france','eng.league_cup','esp.copa_del_rey','ita.coppa_italia','ger.dfb_pokal']
T={}
for f in glob.glob(S+'/u4/4019*.json'):
    d=json.load(open(f))
    for c in d['header']['competitions'][0]['competitors']: T[c['team']['id']]=c['team']['displayName']
jobs=[]
for tid in T:
    for lg in LG:
        try: s=get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{lg}/teams/{tid}/schedule?season=2026")
        except Exception: continue
        for g in s.get('events',[]):
            cc=g['competitions'][0]
            if cc['status']['type'].get('completed') and any(x['team']['id']==tid for x in cc['competitors']): jobs.append((tid,lg,g['id'],g['date']))
sel={}
for tid in T:
    js=sorted({(j[3],j[1],j[2]) for j in jobs if j[0]==tid})[-12:]
    for dt,lg,i in js: sel[i]=lg
def fetch(it):
    i,lg=it; p=f"{S}/u4/sum/{i}.json"
    if os.path.exists(p): return 0
    try: json.dump(get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{lg}/summary?event={i}"),open(p,'w')); return 1
    except Exception: return 0
with ThreadPoolExecutor(12) as ex: n=sum(ex.map(fetch,sel.items()))
json.dump(T,open(S+'/u4/teams.json','w')); print(len(T),'teams',len(sel),'games',n,'downloaded')
