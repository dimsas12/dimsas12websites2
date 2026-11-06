import json,sys,statistics as st
from collections import defaultdict
S=sys.argv[1]; A=json.load(open(S+'/shp/nfl_analysis.json'))
L=[a for a in A if 'line' in a and a['r']['res'] in('W','L')]
def rate(h): return h['hit']/h['n'] if h else None
for a in L:
    ls,cur=a['ls'],a['cur']
    base=ls['avg'] if ls and ls['n']>=6 else (cur['avg'] if cur else None)
    if ls and cur and ls['n']>=6 and cur['n']>=2: base=(ls['avg']*ls['n']*0.5+cur['avg']*cur['n'])/(ls['n']*0.5+cur['n'])
    a['base']=base
    a['mg']=None if base is None else ((base-a['line'])/a['line'] if a['side']=='o' else (a['line']-base)/a['line'])
    if not cur and ls: k='Week 1: last-season baseline'
    elif not ls or ls['n']<6: k='Rookie / new role, current season only'
    elif rate(cur)>=0.6 and rate(ls)<0.5: k='FOLLOW new role (hot this season, weak last)'
    elif rate(cur)<0.5 and rate(ls)>=0.55: k='BUY-LOW (cold start, strong last season)'
    else: k='Consistent'
    a['cat']=k
def sm(lab,v):
    w=sum(x['r']['res']=='W' for x in v); u=sum(x['r']['won'] for x in v)
    print(f"{lab:52s} n={len(v):3d} {w}-{len(v)-w} {w/max(1,len(v)):.0%} units {u:+.2f}")
g=defaultdict(list)
for a in L: g[a['cat']].append(a)
for k,v in g.items(): sm(k,v)
sm('ALL line props',L)
for lab,f in [('margin >=15%',lambda a:a['mg'] is not None and a['mg']>=0.15),('margin 0-15%',lambda a:a['mg'] is not None and 0<=a['mg']<0.15),('margin <0',lambda a:a['mg'] is not None and a['mg']<0)]: sm(lab,[a for a in L if f(a)])
for lab,f in [('LS hit>=60%',lambda a:a['ls'] and a['ls']['n']>=6 and rate(a['ls'])>=0.6),('LS hit 45-59%',lambda a:a['ls'] and a['ls']['n']>=6 and 0.45<=rate(a['ls'])<0.6),('LS hit<45%',lambda a:a['ls'] and a['ls']['n']>=6 and rate(a['ls'])<0.45)]: sm(lab,[a for a in L if f(a)])
sm('teammate target-earner OUT',[a for a in L if a.get('out_mates')])
rush=[a for a in L if a['mk'] in('rushing yards','rush attempts','rushing attempts')]
sm('rush overs, team favoured (spread<0)',[a for a in rush if a.get('spread') is not None and a['spread']<0 and a['side']=='o'])
sm('rush overs, team underdog',[a for a in rush if a.get('spread') is not None and a['spread']>0 and a['side']=='o'])
rec=[a for a in L if a['mk']=='receiving yards']
sm('rec yds, high total (>=47)',[a for a in rec if (a.get('total') or 0)>=47])
sm('rec yds, low total (<47)',[a for a in rec if a.get('total') and a['total']<47])
sm('rec yds, avg targets >=6 (pre-game)',[a for a in rec if (a['cur'] or a['ls']) and (a['cur'] or a['ls'])['tgt']>=6])
sm('rec yds, avg targets <6',[a for a in rec if (a['cur'] or a['ls']) and (a['cur'] or a['ls'])['tgt']<6])
for mk in sorted(set(a['mk'] for a in L)): sm('market '+mk,[a for a in L if a['mk']==mk])
print('median LS hit', st.median([rate(a['ls']) for a in L if a['ls'] and a['ls']['n']>=6]), 'median margin', st.median([a['mg'] for a in L if a['mg'] is not None]))
json.dump(A,open(S+'/shp/nfl_feat.json','w'),default=str)
