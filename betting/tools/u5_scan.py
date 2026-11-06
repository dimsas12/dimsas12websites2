import json,re,html
from collections import defaultdict
E=json.load(open('u5/edges.json'))
GR={'AEK Athens v Real Madrid':'ΑΕΚ – Ρεάλ Μαδρίτης (19:45)','Fenerbahce v Liverpool':'Φενέρμπαχτσε – Λίβερπουλ (19:45)','Borussia Dortmund v Real Betis':'Ντόρτμουντ – Μπέτις (22:00)','FC Porto v Napoli':'Πόρτο – Νάπολι (22:00)','Lens v Como':'Λανς – Κόμο (22:00)','PSV Eindhoven v Club Brugge':'PSV – Μπριζ (22:00)','RB Leipzig v Manchester City':'Λειψία – Μάντσεστερ Σίτι (22:00)','Slavia Prague v Arsenal':'Σλάβια Πράγας – Άρσεναλ (22:00)','Viking FK v Sabah FK':'Βίκινγκ – Σαμπάχ (22:00)'}
MK={'Shots':'Σουτ','Shots on Target':'Σουτ στην εστία','Assists':'Ασίστ','Offsides':'Οφσάιντ','Goalkeeper Saves':'Αποκρούσεις','Fouls':'Φάουλ (αμφίσημο)','Fouls Committed':'Φάουλ που κάνει'}
by=defaultdict(lambda: defaultdict(list))
for r in E: by[r['game']][r['player']].append(r)
out=[]
for g,pl in by.items():
    nl=sum(len(v) for v in pl.values())
    out.append(f'<details><summary>{g} · {len(pl)} παίκτες · {nl} γραμμές DraftKings</summary><div class="tbl"><table><tr><th>Παίκτης</th><th>Βασικός (τελ. 10)</th><th>Καλύτερη γραμμή στο εύρος</th><th>Τελευταίες τιμές</th><th>Hit rate</th><th>Εκτ. πιθ.</th><th>EV</th><th>Απόφαση</th></tr>')
    rows=[]
    for p,v in pl.items():
        inr=[r for r in v if 1.65<=r['dec']<=2.51]
        best=max(inr,key=lambda r:r['ev']) if inr else max(v,key=lambda r:r['ev'])
        dec=('φάουλ: αμφίσημη αγορά' if 'Fouls' in best['mk'] else ('εκτός εύρους αποδόσεων' if not inr else ('value, χωρίς αναλυτική θέση (ζ) / βλ. «Κοντά»' if best['ev']>0.08 else 'χωρίς αξία')))
        rows.append((-best['ev'],f"<tr><td>{html.escape(p)}</td><td>{best['allstarts']}</td><td>{MK.get(best['mk'],best['mk'])} {int(best['tgt'])}+ {best['am']} ({best['dec']:.2f})</td><td>{', '.join(map(str,best['vals']))}</td><td>{best['hits']}</td><td>{best['est']:.0%}</td><td>{best['ev']:+.0%}</td><td class=\"{'y' if dec=='ΕΠΙΛΟΓΗ' else ''}\">{dec}</td></tr>"))
    out+= [r[1] for r in sorted(rows)]
    out.append('</table></div></details>')
open('u5/scan_props.html','w').write('\n'.join(out))
print(len(E),'lines',sum(len(p) for p in by.values()),'players')
