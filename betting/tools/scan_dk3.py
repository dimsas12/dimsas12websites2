"""NBA scan tables with DraftKings prop lines (scoresandodds) for every rotation player of tonight's games."""
import json, statistics as st, html, sys
players = json.load(open(sys.argv[1])); edges = json.load(open('dk/dk_edges.json')); sb = json.load(open(sys.argv[2]))
GAMES = sys.argv[3].split(','); PICK = {("BOS", "Paul George"): "ΕΠΙΛΟΓΗ: O7.5 Ρ+Α"}
ABBR = {'GSW': 'GS'}
M = {'pts': 'Π', 'reb': 'Ρ', 'ast': 'Α', 'tpm': '3P', 'pra': 'PRA', 'pr': 'Π+Ρ', 'pa': 'Π+Α', 'ra': 'Ρ+Α'}
by = {}
for e in edges:
    if e.get('n'): by.setdefault((e['team'], e['player']), []).append(e)
out = []
for ev in sb['events']:
    c = ev['competitions'][0]
    h = [x for x in c['competitors'] if x['homeAway'] == 'home'][0]['team']['abbreviation']; a = [x for x in c['competitors'] if x['homeAway'] == 'away'][0]['team']['abbreviation']
    if f"{a}@{h}" not in GAMES: continue
    o = (c.get('odds') or [{}])[0]
    rows = []
    for t in (a, h):
        pl = sorted([(k.split('|')[1], v) for k, v in players.items() if k.split('|')[0] == t], key=lambda kv: -st.mean(x['min'] for x in kv[1]))
        for name, v in pl:
            m = lambda k: st.mean(x[k] for x in v)
            es = sorted(by.get((t, name), []), key=lambda e: -e['ev'])
            lines = '; '.join(f"{M[e['market']]} {e['side'].upper()}{e['line']:g} {e['american']:+d} ({e['hits']})" for e in es[:4]) or 'χωρίς γραμμή DraftKings (στην πηγή)'
            best = f"{es[0]['side'].upper()}{es[0]['line']:g} {M[es[0]['market']]} ({es[0]['ev']:+.0%})" if es else '–'
            pk = PICK.get((t, name))
            if pk: dec, cls = pk, 'y'
            elif es and es[0]['ev'] > 0.1 and es[0]['side'] == 'u': dec, cls = 'under: απορρίφθηκε (κανόνας under/ρόλος)', ''
            elif es and es[0]['ev'] > 0.1: dec, cls = 'value αλλά απορρίφθηκε (βλ. «Κοντά»)', ''
            elif es: dec, cls = 'χωρίς αξία', 'n'
            else: dec, cls = '—', ''
            pts = ', '.join(str(x['pts']) for x in v[-5:])
            rows.append(f"<tr><td>{t}</td><td>{html.escape(name)}</td><td>{len(v)}</td><td>{m('min'):.0f}</td><td>{m('pts'):.1f}</td><td>{m('reb'):.1f}</td><td>{m('ast'):.1f}</td><td>{m('tpm'):.1f}</td><td>{pts}</td><td>{lines}</td><td>{best}</td><td class=\"{cls}\">{dec}</td></tr>")
    out.append(f"<details><summary>{a} @ {h} · DraftKings: {o.get('details')} · O/U {o.get('overUnder')}</summary>\n<div class=\"tbl\"><table><tr><th>Ομάδα</th><th>Παίκτης</th><th>GP</th><th>Λεπτά</th><th>Π</th><th>Ρ</th><th>Α</th><th>3P</th><th>Πόντοι (τελ. 5)</th><th>Γραμμές DraftKings (hit rate)</th><th>Καλύτερη πλευρά (EV)</th><th>Απόφαση</th></tr>\n" + '\n'.join(rows) + "</table></div></details>")
open('scan_dk03.html', 'w').write('\n'.join(out)); print(len(out), sum(x.count('<tr>')-1 for x in out))
