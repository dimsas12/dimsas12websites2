import json,sys,urllib.request,gzip
from datetime import date,timedelta
S=sys.argv[1]; UA={"User-Agent":"Mozilla/5.0","Accept-Encoding":"gzip"}
def get(u):
    b=urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=40).read()
    if b[:2]==b'\x1f\x8b': b=gzip.decompress(b)
    return json.loads(b)
M={}
d=date(2026,7,20)
while d<=date(2026,11,2):
    for lg in ('atp','wta'):
        try: sb=get(f"https://site.api.espn.com/apis/site/v2/sports/tennis/{lg}/scoreboard?dates={d:%Y%m%d}")
        except Exception as e: continue
        for ev in sb.get('events',[]):
            for g in ev.get('groupings',[]):
                if 'Singles' not in g['grouping']['displayName']: continue
                for c in g['competitions']:
                    if not c['status']['type'].get('completed'): continue
                    comp=[]
                    for x in c['competitors']:
                        comp.append(dict(id=x['id'],name=x['athlete']['displayName'],win=x.get('winner'),seed=x.get('seed') or x.get('curatedRank',{}).get('current'),ls=[l.get('value') for l in x.get('linescores',[])]))
                    M[c['id']]=dict(lg=lg,tour=ev['name'],date=c['date'],round=c.get('round',{}).get('displayName'),note=(c.get('notes') or [{}])[0].get('text',''),comp=comp,court=c.get('venue',{}).get('court',''))
    d+=timedelta(days=3)
json.dump(M,open(S+'/shp/ten/matches.json','w')); print(len(M),'matches')
k=list(M)[0]; print(M[k])
