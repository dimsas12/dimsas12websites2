import json,statistics as st,sys
R=json.load(open(sys.argv[1]+'/core_edges.json'))
def am2d(a):
    a=int(a); return 1+(a/100 if a>0 else 100/-a)
out=[]
for r in R:
    if not (1.65<=r['dec']<=2.51) or r['ncur']<4: continue
    hc,nc=map(int,r['hc'].split('/')); mc=st.mean(r['last']); ln=r['tgt']-0.5
    if hc/nc<0.6 or r['big']<0.58 or (mc-ln)/ln<0.08: continue
    try: mv=am2d(r['open'])-r['dec']
    except: mv=0
    r['curmean']=round(mc,1); r['move']=round(mv,2); out.append(r)
json.dump(out,open(sys.argv[1]+'/cands.json','w'))
print(len(out))
for r in sorted(out,key=lambda x:-x['ev'])[:int(sys.argv[2]) if len(sys.argv)>2 else 40]:
    mv=r['move']
    print(f"{r['game']:10s} {r['player'][:20]:20s} {r['mk']:4s} {int(r['tgt'])}+ {r['am']:>5s} op {r['open']:>5s} {'↓' if mv>0.03 else ('↑' if mv<-0.03 else '=')} | cur {r['hc']} ls {r['hl']} big {r['big']} l6 {r['curmean']} | est {r['est']:.2f} EV {r['ev']:+.2f} {r['last']} min {r['mins']}")
