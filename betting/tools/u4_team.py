import json,glob,sys
S=sys.argv[1]; T=json.load(open(S+'/u4/teams.json'))
rows={}
for f in glob.glob(S+'/u4/sum/*.json'):
    d=json.load(open(f)); c=d['header']['competitions'][0]
    bx={t['team']['displayName']:{s['name']:s.get('displayValue') for s in t.get('statistics',[])} for t in d.get('boxscore',{}).get('teams',[])}
    cs={x['team']['displayName']:(x['homeAway'],x.get('score')) for x in c['competitors']}
    lg=d['header'].get('league',{}).get('abbreviation') or ''
    for nm,(ha,sc) in cs.items():
        opp=[k for k in cs if k!=nm][0]
        rows.setdefault(nm,[]).append((c['date'][:10],lg[:6],ha[0],opp[:16],sc,cs[opp][1],bx.get(nm,{}).get('shotsOnTarget'),bx.get(opp,{}).get('shotsOnTarget'),bx.get(nm,{}).get('totalShots'),bx.get(opp,{}).get('totalShots'),bx.get(nm,{}).get('wonCorners'),bx.get(opp,{}).get('wonCorners'),bx.get(nm,{}).get('possessionPct')))
for t in sys.argv[2:]:
    print('==',t,'(date lg h/a opp GF-GA SOTfor-SOTag shots for-ag corners for-ag poss)')
    for r in sorted(rows.get(t,[]))[-12:]: print('  ',r)
