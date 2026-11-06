import json,urllib.request,gzip,unicodedata
from datetime import date,timedelta
UA={"User-Agent":"Mozilla/5.0","Accept-Encoding":"gzip"}
def get(u):
    b=urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=40).read()
    if b[:2]==b'\x1f\x8b': b=gzip.decompress(b)
    return json.loads(b)
def norm(s): return unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode().lower()
M={}
d=date(2026,9,1)
while d<=date(2026,11,3):
    try: sb=get(f"https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard?dates={d:%Y%m%d}")
    except Exception: d+=timedelta(days=1); continue
    for ev in sb.get('events',[]):
        for g in ev.get('groupings',[]):
            if 'Singles' not in g['grouping']['displayName']: continue
            for c in g['competitions']:
                if not c['status']['type'].get('completed'): continue
                M[c['id']]=dict(tour=ev['name'],date=c['date'],rd=c.get('round',{}).get('displayName'),note=(c.get('notes') or [{}])[0].get('text',''),
                    comp=[(x['athlete']['displayName'],x.get('winner'),[l.get('value') for l in x.get('linescores',[])]) for x in c['competitors']])
    d+=timedelta(days=4 if d<date(2026,10,25) else 1)
json.dump(M,open('u4/ten.json','w'))
an=json.load(open('an_atp04.json'))
ms=sorted(M.values(),key=lambda m:m['date'])
for c in an['competitions']:
    ps={x['side']:x['player']['full_name'] for x in c['competitors']}
    o=[x for x in c['odds'] if x['book_id']==68 and x['type']=='competition']
    print(f"\n## {ps.get('home')} ({o[0]['ml_home'] if o else ''}) vs {ps.get('away')} ({o[0]['ml_away'] if o else ''}) {c['start_time']} {c['meta'].get('title')}")
    for side in ('home','away'):
        p=norm(ps[side]); w=l=0; seq=[]
        for m in ms:
            for n,win,ls in m['comp']:
                if norm(n)==p:
                    seq.append(f"{m['date'][5:10]} {m['tour'][:12]} {m['rd'][:8] if m['rd'] else ''} {'W' if win else 'L'} {m['note'][:40]}")
                    w+=bool(win); l+=not win
        print(f"  {ps[side]}: {w}-{l} since Sep (sampled)"); 
        for s in seq[-6:]: print('     ',s)
