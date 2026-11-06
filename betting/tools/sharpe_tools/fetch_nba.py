import json,os,sys,urllib.request,gzip
S=sys.argv[1]; UA={"User-Agent":"Mozilla/5.0 Chrome/124","Accept-Encoding":"gzip"}
def get(u):
    r=urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=40); b=r.read()
    if b[:2]==b'\x1f\x8b': b=gzip.decompress(b)
    return json.loads(b)
ids=[]
for d in range(20,33):
    ds=f"202610{d}" if d<=31 else "20261101"
    sb=get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={ds}")
    for e in sb['events']:
        ids.append((ds,e['id']))
n=0
for ds,i in ids:
    p=f"{S}/shp/sum/{i}.json"
    if os.path.exists(p): continue
    json.dump(get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={i}"),open(p,'w')); n+=1
print(len(ids),'games',n,'downloaded')
