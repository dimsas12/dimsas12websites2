import json,sys,html
from collections import defaultdict
S=sys.argv[1]; e=lambda s:html.escape(str(s))
def dec(o): return 1+(o/100 if o>0 else 100/-o)
TH="<tr><th>Group</th><th>Bets</th><th>W-L</th><th>Hit</th><th>Units</th><th>Read</th></tr>"
def row(lab,v,note=''):
    s=[a for a in v if a['r']['res'] in('W','L')]; w=sum(a['r']['res']=='W' for a in s); u=sum(a['r']['won'] for a in s); n=len(s)
    c='pos' if u>0 else 'neg'
    return f"<tr><td>{e(lab)}</td><td class=num>{n}</td><td class=num>{w}-{n-w}</td><td class=num>{w/max(1,n):.0%}</td><td class='num {c}'>{u:+.2f}</td><td>{e(note)}</td></tr>"
A=json.load(open(S+'/shp/nfl_feat.json')); L=[a for a in A if 'line' in a and a['r']['res'] in('W','L')]
def rate(h): return h['hit']/h['n'] if h else None
grp=[('Margin ≥15% (line well inside blended baseline)',[a for a in L if a.get('mg') is not None and a['mg']>=0.15],'Where his NFL profit is.'),
     ('Margin 0–15%',[a for a in L if a.get('mg') is not None and 0<=a['mg']<0.15],'Break-even or worse.'),
     ('Line above baseline (margin <0)',[a for a in L if a.get('mg') is not None and a['mg']<0],'Bets on a new role the numbers did not show yet.'),
     ('Last-season hit rate ≥60%',[a for a in L if a['ls'] and a['ls']['n']>=6 and rate(a['ls'])>=0.6],'Long sample clearly over the line.'),
     ('Last-season hit rate 45–59%',[a for a in L if a['ls'] and a['ls']['n']>=6 and 0.45<=rate(a['ls'])<0.6],'Coin-flip history loses.'),
     ('Week 1, last-season baseline',[a for a in L if a.get('cat','').startswith('Week 1')],'Opening lines, same as his NBA opening week.'),
     ('Followed a new role (hot this year, weak last year)',[a for a in L if a.get('cat','').startswith('FOLLOW')],'In the NFL roles stick, and books are slow.'),
     ('Buy-low on a cold start',[a for a in L if a.get('cat','').startswith('BUY')],'Loses here too, like in the NBA.'),
     ('Teammate target-earner out',[a for a in L if a.get('out_mates')],'Not an edge, same as the NBA.'),
     ('Rushing overs, team favoured',[a for a in L if a['mk'] in('rushing yards','rush attempts','rushing attempts') and a['side']=='o' and a.get('spread') is not None and a['spread']<0],'Game script: leading teams run.'),
     ('Rushing overs, team underdog',[a for a in L if a['mk'] in('rushing yards','rush attempts','rushing attempts') and a['side']=='o' and a.get('spread') is not None and a['spread']>0],'Trailing teams pass.'),
     ('Receiving yards, game total ≥47',[a for a in L if a['mk']=='receiving yards' and (a.get('total') or 0)>=47],'Shootout games.'),
     ('Receiving yards, game total <47',[a for a in L if a['mk']=='receiving yards' and a.get('total') and a['total']<47],'')]
nfl_grp=''.join(row(l,v,n) for l,v,n in grp)
def hs(h): return f"{h['hit']}/{h['n']} · {h['avg']:g}" if h else '—'
rows=[]
for a in A:
    r=a['r']; c={'W':'pos','L':'neg'}.get(r['res'],'')
    if 'line' not in a:
        rows.append(f"<tr><td>{r['date'][5:]}</td><td>{e(r['who'])}</td><td>{e(r['prop'])}</td><td class=num>{(r['odds'] or 0):+.0f}</td><td colspan=7>{e(a.get('note',''))}</td><td class='{c}'>{r['res']}</td></tr>"); continue
    cur=a['cur']; vol=cur or a['ls']; vs=f"tgt {vol['tgt']:g} · car {vol['car']:g}" if vol else '—'
    mg=a.get('mg'); mgs=f"{mg:+.0%}" if mg is not None else '—'
    sp=a.get('spread'); g=f"{sp:+g} / {a.get('total')}" if sp is not None else '—'
    act=a.get('actual'); acts='DNP' if act is None else f"{act:g}"
    rows.append(f"<tr><td>{r['date'][5:]}</td><td>{e(a['name'])}</td><td>{e(r['prop'])}</td><td class=num>{r['odds']:+.0f}</td><td class=num>{hs(a['ls'])}</td><td class=num>{hs(cur)}</td><td class=num>{vs}</td><td class=num>{mgs}</td><td class=num>{g}</td><td>{e(a.get('cat','—'))}{' · out: '+e(', '.join(a['out_mates'])) if a.get('out_mates') else ''}</td><td class=num>{acts}</td><td class='{c}'>{r['res']}</td></tr>")
nfl_rows=''.join(rows)
T=json.load(open(S+'/shp/ten_analysis.json'))
ok=[a for a in T if 'tour' in a and a['r']['res'] in('W','L') and a['won']==(a['r']['res']=='W') and a['r']['prop']=='Moneyline']
def p(x): w,l,_=x; return w/(w+l) if w+l else 0.5
tg=[('Opponent played more matches this week',[a for a in ok if sum(a['op_week'][:2])>sum(a['me_week'][:2])],'Fresher player against a tired one.'),
    ('Early rounds (qualifying, R1, R2)',[a for a in ok if any(k in a['round'] for k in ('Round 1','Round 2','Qualifying'))],'Bigger gaps in level, softer lines.'),
    ('Late rounds (R3 to final)',[a for a in ok if not any(k in a['round'] for k in ('Round 1','Round 2','Qualifying'))],''),
    ('Better 60-day record than opponent',[a for a in ok if p(a['me_rec'])>p(a['op_rec'])],''),
    ('Worse or equal 60-day record',[a for a in ok if p(a['me_rec'])<=p(a['op_rec'])],'Form alone did not decide it.'),
    ('Favourite (−105 to −178)',[a for a in ok if a['r']['odds']<0],'The core of his tennis.'),
    ('Underdog (+125 to +150)',[a for a in ok if a['r']['odds']>0],'Rare, vs big names.')]
ten_grp=''.join(row(l,v,n) for l,v,n in tg)
tr=[]
for a in T:
    r=a['r']; c={'W':'pos','L':'neg'}.get(r['res'],'')
    if 'tour' not in a: tr.append(f"<tr><td>{r['date'][5:]}</td><td>{e(r['who'])}</td><td>{e(r['prop'])}</td><td class=num>{r['odds']:+.0f}</td><td colspan=5>match not found in ESPN draws</td><td class='{c}'>{r['res']}</td></tr>"); continue
    flag='' if a['won']==(r['res']=='W') or r['prop']!='Moneyline' else ' (date may point to another match)'
    m,o=a['me_rec'],a['op_rec']
    tr.append(f"<tr><td>{r['date'][5:]}</td><td>{e(r['who'])}</td><td>{e(r['prop'])}</td><td class=num>{r['odds']:+.0f}</td><td>{e(a['tour'])} · {e(a['round'])}</td><td>{e(a['opp'])}</td><td class=num>{m[0]}-{m[1]} {m[2]} / {o[0]}-{o[1]} {o[2]}</td><td class=num>{a['me_week'][0]+a['me_week'][1]} / {a['op_week'][0]+a['op_week'][1]}</td><td>{e(a['score'])}{e(flag)}</td><td class='{c}'>{r['res']}</td></tr>")
ten_rows=''.join(tr)
p_=S+'/sharpe_tools/tpl.html'; t=open(p_).read()
add=f'''
<section>
  <h2>NFL: what kind of spot he bet</h2>
  <p class="note">Baseline = last season's average blended with this season's games (this season weighted double). 129 line props settled.</p>
  <div class="tw"><table>{TH}{nfl_grp}</table></div>
</section>
<section>
  <h2>NFL, pick by pick, with the numbers he could see</h2>
  <p class="note">Last season and this season: games over/under his line · average. Volume: targets and carries per game before the pick. Spread / total: the DraftKings line for his player's team from ESPN.</p>
  <div class="tw"><table><tr><th>Date</th><th>Player</th><th>Prop</th><th>Odds</th><th>Last season</th><th>This season</th><th>Volume</th><th>Margin</th><th>Spread / total</th><th>Spot</th><th>Actual</th><th>Res</th></tr>{nfl_rows}</table></div>
</section>
<section>
  <h2>Tennis: what kind of spot he bet</h2>
  <p class="note">From ESPN draws: each player's record in the 60 days before the match, matches already played that week, round and result. 23 moneyline bets could be matched cleanly.</p>
  <div class="tw"><table>{TH}{ten_grp}</table></div>
  <div class="tw"><table><tr><th>Date</th><th>Player</th><th>Bet</th><th>Odds</th><th>Event</th><th>Opponent</th><th>60-day record (his / opp)</th><th>Matches this week</th><th>Result</th><th>Res</th></tr>{ten_rows}</table></div>
</section>
'''
marker='<section>\n  <h2>Full ledger, all 390 bets</h2>'
if 'NFL: what kind of spot he bet' not in t: t=t.replace(marker,add+marker)
else:
    import re
    t=re.sub(r'\n<section>\n  <h2>NFL: what kind of spot he bet</h2>.*?(?=<section>\n  <h2>Full ledger)',add.lstrip('\n')+'\n',t,flags=re.S) if False else t
open(S+'/sharpe_tools/tpl2.html','w').write(t); print('ok', len(t))
