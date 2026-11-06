import json,sys,re,unicodedata
from datetime import datetime,timedelta
S=sys.argv[1]; M=json.load(open(S+'/shp/ten/matches.json'))
R=[r for r in json.load(open(S+'/sharpe.json'))['rows'] if r['sport']=='Tennis']
def norm(s): return unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode().lower().strip()
ms=sorted(M.values(),key=lambda m:m['date'])
def rec(name,before,days=60):
    lo=(before-timedelta(days=days)).isoformat()
    w=l=0; last=[]
    for m in ms:
        if not (lo<=m['date'][:19]<before.isoformat()): continue
        for c in m['comp']:
            if norm(c['name'])==norm(name):
                if c['win']: w+=1
                else: l+=1
                last.append('W' if c['win'] else 'L')
    return w,l,''.join(last[-6:])
out=[]
for r in R:
    d0=datetime.fromisoformat(r['date']); lo=(d0-timedelta(hours=14)).isoformat(); hi=(d0+timedelta(hours=38)).isoformat()
    cand=[m for m in ms if lo<=m['date'][:19]<=hi and any(norm(c['name'])==norm(r['who']) for c in m['comp'])]
    info=dict(r=r)
    if cand:
        m=cand[0]; me=[c for c in m['comp'] if norm(c['name'])==norm(r['who'])][0]; op=[c for c in m['comp'] if c is not me][0]
        t=datetime.fromisoformat(m['date'][:19])
        info.update(tour=m['tour'],round=m['round'],opp=op['name'],won=me['win'],score=m['note'],me_rec=rec(me['name'],t),op_rec=rec(op['name'],t),
                    h2h=sum(1 for x in ms if x['date']<m['date'] and {norm(c['name']) for c in x['comp']}=={norm(me['name']),norm(op['name'])}),
                    me_week=rec(me['name'],t,6),op_week=rec(op['name'],t,6))
    out.append(info)
json.dump(out,open(S+'/shp/ten_analysis.json','w'),default=str)
for i in out:
    r=i['r']
    if 'tour' not in i: print(r['date'],r['who'],r['prop'],r['odds'],r['res'],'| match not found'); continue
    a,b=i['me_rec'],i['op_rec']
    print(f"{r['date']} {r['res']} {r['who']:22s} {r['prop']:18s} {r['odds']:+.0f} | {i['tour'][:28]:28s} {i['round'][:14]:14s} vs {i['opp']:22s} | 60d {a[0]}-{a[1]} {a[2]:6s} vs {b[0]}-{b[1]} {b[2]:6s} | thisweek {i['me_week'][0]}-{i['me_week'][1]} vs {i['op_week'][0]}-{i['op_week'][1]} | {i['score'][:60]}")
