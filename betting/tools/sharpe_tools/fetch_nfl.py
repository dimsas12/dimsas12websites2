import json,os,sys,urllib.request,gzip
S=sys.argv[1]; UA={"User-Agent":"Mozilla/5.0 Chrome/124","Accept-Encoding":"gzip"}
def get(u):
    b=urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=40).read()
    if b[:2]==b'\x1f\x8b': b=gzip.decompress(b)
    return json.loads(b)
ids=[]
for w in range(1,10):
    sb=get(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?week={w}&seasontype=2&dates=2026")
    for e in sb['events']: ids.append((w,e['id'],e['date'],e['status']['type']['completed']))
n=0
for w,i,d,c in ids:
    p=f"{S}/shp/nfl/{i}.json"
    if not c or os.path.exists(p): continue
    json.dump(get(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={i}"),open(p,'w')); n+=1
print(len(ids),'games',sum(c for *_,c in ids),'completed',n,'downloaded')
