import json,sys
from collections import defaultdict
S=sys.argv[1]; A=[a for a in json.load(open(S+'/shp/nba_analysis.json')) if 'line' in a and a['r']['res'] in('W','L')]
def dec(o): return 1+(o/100 if o>0 else 100/-o)
def rate(h): return h['hit']/h['n'] if h else None
grp=defaultdict(list)
for a in A:
    ls,l15,cur=rate(a['ls']),rate(a['ls15']),rate(a['cur'])
    base=a['ls']['avg'] if a['ls'] else (a['cur']['avg'] if a['cur'] else None)
    mg=None if base is None else ((base-a['line'])/a['line'] if a['side']=='o' else (a['line']-base)/a['line'])
    a['mg']=mg
    if not a['ls'] and (not a['cur'] or a['cur']['n']<2): k='rookie/no history'
    elif a['ls'] and cur is not None and a['cur']['n']>=2 and cur<0.5 and ls>=0.5: k='BUY-LOW: weak start, strong last season'
    elif cur is not None and a['cur']['n']>=2 and cur>=0.6 and (ls is None or ls<0.55): k='FOLLOW new role: hot start, weak last season'
    elif a['cur'] is None or a['cur']['n']<2: k='opening week: last-season baseline only'
    else: k='consistent (both >=50%)'
    grp[k].append(a)
    a['cat']=k
def sm(lab,v):
    w=sum(x['r']['res']=='W' for x in v); u=sum(x['r']['won'] for x in v)
    print(f"{lab:48s} n={len(v):2d} {w}-{len(v)-w} units {u:+.2f}")
for k,v in grp.items(): sm(k,v)
sm('ALL NBA',A)
for lab,f in [('line below LS-avg (margin>=10%)',lambda a:a['mg'] is not None and a['mg']>=0.10),('margin 0-10%',lambda a:a['mg'] is not None and 0<=a['mg']<0.10),('margin <0 (line above baseline)',lambda a:a['mg'] is not None and a['mg']<0)]:
    sm(lab,[a for a in A if f(a)])
for lab,f in [('LS hit>=60%',lambda a:a['ls'] and rate(a['ls'])>=0.6),('LS hit 50-59%',lambda a:a['ls'] and 0.5<=rate(a['ls'])<0.6),('LS hit <50%',lambda a:a['ls'] and rate(a['ls'])<0.5)]:
    sm(lab,[a for a in A if f(a)])
sm('teammate starter OUT (role up)',[a for a in A if a.get('out_mates')])
sm('unders',[a for a in A if a['side']=='u'])
json.dump(A,open(S+'/shp/nba_feat.json','w'),default=str)
import statistics as st
print('median LS hit rate of picks', st.median([rate(a['ls']) for a in A if a['ls']]))
print('median margin', st.median([a['mg'] for a in A if a['mg'] is not None]))
