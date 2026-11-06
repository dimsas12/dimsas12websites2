"""Blend Poisson model 1:2 with de-vigged market for DNB, O/U 2.5/3.5 and BTTS; print EV>=threshold, flag unders."""
import re, sys
txt = open(sys.argv[1]).read().split('### ')[1:]
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
for b in txt:
    title = b.splitlines()[0].split('  xG')[0]
    m = re.search(r'model 1/X/2 ([\d.]+)/([\d.]+)/([\d.]+)\s+U2.5 ([\d.]+) U3.5 ([\d.]+) BTTS ([\d.]+)', b)
    if not m: continue
    p1, px, p2, u25, u35, bt = map(float, m.groups())
    out = []
    def blend(pm, oa, ob, labels):
        fa = (1/oa)/(1/oa+1/ob)
        pa = (pm + 2*fa)/3
        for lab, p, o in ((labels[0], pa, oa), (labels[1], 1-pa, ob)):
            ev = p*o-1
            if ev >= thr: out.append(f"{lab} @{o} p{p:.2f} EV{ev:+.2f}")
    d = re.search(r"VB DNB \('([\d.]+)', '([\d.]+)'\)", b)
    if d: blend(p1/(p1+p2), float(d.group(1)), float(d.group(2)), ('DNB-1', 'DNB-2'))
    o = re.search(r"VB O/U2.5 \('([\d.]+)', '([\d.]+)'\)", b)
    if o: blend(1-u25, float(o.group(1)), float(o.group(2)), ('O2.5', 'U2.5*'))
    o = re.search(r"VB O/U3.5 \('([\d.]+)', '([\d.]+)'\)", b)
    if o: blend(1-u35, float(o.group(1)), float(o.group(2)), ('O3.5', 'U3.5*'))
    o = re.search(r"VB BTTS \('([\d.]+)', '([\d.]+)'\)", b)
    if o: blend(bt, float(o.group(1)), float(o.group(2)), ('BTTS-Y', 'BTTS-N*'))
    print(f"{title:32s} | " + ' ; '.join(out))
