import json,glob,math,re,sys
S=sys.argv[1]
def dec(a): a=float(a); return 1+(a/100 if a>0 else 100/-a)
txt=open(S+'/u6/scan.txt').read()
blocks=txt.split('### ')[1:]
def pois(l,k): return math.exp(-l)*l**k/math.factorial(k)
for b in blocks:
    hd=b.split('\n')[0]
    if 'ML None' in hd: continue
    m=re.search(r'ML (\S+) (\S+) (\S+)',hd); ou=re.search(r'O/U (\S+) (\S+) (\S+)',hd)
    h,dr,a=[dec(x) for x in m.groups()]; s=1/h+1/dr+1/a; ph,pd,pa=1/h/s,1/dr/s,1/a/s
    rows=re.findall(r'(home|away)\s+(.+?)\s+last\d+ GF ([\d.]+) GA ([\d.]+)',b)
    t={r[0]:(r[1].strip(),float(r[2]),float(r[3])) for r in rows}
    L=1.35; sh=lambda x:1+0.5*(x/L-1)
    lh=L*sh(t['home'][1])*sh(t['away'][2])*1.1; la=L*sh(t['away'][1])*sh(t['home'][2])*0.9
    P=[[pois(lh,i)*pois(la,j) for j in range(10)] for i in range(10)]
    mh=sum(P[i][j] for i in range(10) for j in range(10) if i>j); md=sum(P[i][i] for i in range(10)); ma=1-mh-md
    bl=lambda x,y:(x+2*y)/3
    o25=sum(P[i][j] for i in range(10) for j in range(10) if i+j>2); o35=sum(P[i][j] for i in range(10) for j in range(10) if i+j>3)
    btts=sum(P[i][j] for i in range(1,10) for j in range(1,10))
    print(f"{hd[:11]} {t['home'][0]:18s} v {t['away'][0]:18s} lam {lh:.2f}-{la:.2f} | mkt H{ph:.2f} D{pd:.2f} A{pa:.2f} | mdl H{mh:.2f} D{md:.2f} A{ma:.2f} | bl H{bl(mh,ph):.2f} A{bl(ma,pa):.2f} 1X{bl(mh+md,ph+pd):.2f} X2{bl(ma+md,pa+pd):.2f} | O2.5 {o25:.2f} O3.5 {o35:.2f} BTTS {btts:.2f} | DK O/U {ou.groups()}")
