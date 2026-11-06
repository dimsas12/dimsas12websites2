import json,urllib.request,gzip,sys,os,statistics as st
from concurrent.futures import ThreadPoolExecutor
B=sys.argv[1]; UA={"User-Agent":"Mozilla/5.0","Accept-Encoding":"gzip"}
def get(u):
    b=urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=40).read()
    if b[:2]==b'\x1f\x8b': b=gzip.decompress(b)
    return json.loads(b)
P=json.load(open(B+'/props.json')); L=json.load(open(B+'/logs.json'))
M={'Receiving Yards Milestones':('receiving.receivingYards','receivingYards'),'Rushing Yards Milestones':('rushing.rushingYards','rushingYards'),'Rushing Attempts Milestones':('rushing.rushingAttempts','rushingAttempts'),'Passing Attempts Milestones':('passing.completions/passingAttempts','passingAttempts'),'Passing Yards Milestones':('passing.passingYards','passingYards'),'Receptions Milestones':('receiving.receptions','receptions'),'Rushing + Receiving Yards Milestones':('rr','rr')}
def num(x):
    try: return float(str(x).split('/')[-1])
    except: return 0.0
def cur(aid,k):
    out=[]
    for g in L.get(aid,[]):
        if k=='rr': out.append(num(g.get('rushing.rushingYards'))+num(g.get('receiving.receivingYards')))
        else:
            if k not in g and not (k.startswith('receiving') and any(x.startswith('receiving') for x in g)) and not (k.startswith('rushing') and any(x.startswith('rushing') for x in g)): out.append(0.0); continue
            out.append(num(g.get(k,0)))
    return out
def last(aid):
    p=f"{B}/gl/{aid}.json"
    if not os.path.exists(p):
        try: json.dump(get(f"https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{aid}/gamelog?season=2025"),open(p,'w'))
        except Exception as e: json.dump({},open(p,'w'))
    d=json.load(open(p)); nm=d.get('names',[]); G=[]
    for stp in d.get('seasonTypes',[]):
        if 'Regular' not in stp.get('displayName',''): continue
        for c in stp['categories']:
            for e in c.get('events',[]): G.append(dict(zip(nm,e['stats'])))
    return G
aids={p['aid'] for p in P if p['aid']}
with ThreadPoolExecutor(10) as ex: LS=dict(zip(aids,ex.map(last,aids)))
rows=[]
for p in P:
    if p['type'] not in M or not p.get('odds'): continue
    try: dec=float(p['odds']['decimal']['value']); am=p['odds']['american']['value']; op=p['odds']['american'].get('open')
    except: continue
    t=p['raw']['target']['value']; ck,lk=M[p['type']]
    cs=cur(p['aid'],ck)
    if lk=='rr': ls=[num(g.get('rushingYards'))+num(g.get('receivingYards')) for g in LS.get(p['aid'],[])]
    else: ls=[num(g.get(lk,0)) for g in LS.get(p['aid'],[]) if lk in g]
    if not cs: continue
    hc=sum(x>=t for x in cs); hl=sum(x>=t for x in ls)
    base=(2*sum(cs)+sum(ls))/(2*len(cs)+len(ls)) if ls else st.mean(cs)
    rows.append(dict(player=p['player'],mk=p['type'].replace(' Milestones',''),tgt=t,am=am,open=op,dec=dec,cur=cs,hc=f"{hc}/{len(cs)}",hl=f"{hl}/{len(ls)}",base=round(base,1),margin=round((base-t)/t,2) if t else 0,big=round((hc+hl)/(len(cs)+len(ls)),2)))
json.dump(rows,open(B+'/edges.json','w'))
# keep the full-game ladder: split by restart like NBA
ok=[r for r in rows if 1.65<=r['dec']<=2.51]
print(len(rows),'lines',len(ok),'in range')
for r in sorted(ok,key=lambda r:-(r['big']*r['dec'])):
    print(f"{r['player'][:18]:18s} {r['mk'][:22]:22s} {r['tgt']:>5}+ {r['am']:>5} op {str(r['open']):>5} | cur {r['hc']} {[int(x) for x in r['cur']]} ls {r['hl']} big {r['big']} base {r['base']} marg {r['margin']:+.2f} | bigxdec {r['big']*r['dec']:.2f}")
