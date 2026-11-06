import json,statistics as st,html,sys
base=sys.argv[1]; PL=json.load(open(base+'/players.json')); E=json.load(open(base+'/core_edges.json')); sb=json.load(open(base+'/sb.json'))
PICK={('PHI','Tyrese Maxey'):'ΕΠΙΛΟΓΗ: Π+Α 30+','PHI|VJ Edgecombe':None,('PHI','VJ Edgecombe'):'ΕΠΙΛΟΓΗ: Ρ+Α 8+'}
MK={'pts':'Π','reb':'Ρ','ast':'Α','tpm':'3P','pra':'PRA','pr':'Π+Ρ','pa':'Π+Α','ra':'Ρ+Α'}
by={}
for e in E: by.setdefault(e['player'],[]).append(e)
out=[]; nplayers=0
for ev in sb['events']:
    c=ev['competitions'][0]; o=(c.get('odds') or [{}])[0]
    h=[x for x in c['competitors'] if x['homeAway']=='home'][0]['team']['abbreviation']; a=[x for x in c['competitors'] if x['homeAway']=='away'][0]['team']['abbreviation']
    rows=[]
    for t in (a,h):
        pl=sorted([(k.split('|')[1],v) for k,v in PL.items() if k.split('|')[0]==t],key=lambda kv:-st.mean(x['min'] for x in kv[1]))
        for name,v in pl:
            if st.mean(x['min'] for x in v)<10: continue
            nplayers+=1
            m=lambda k: st.mean(x[k] for x in v)
            es=[e for e in by.get(name,[]) if 1.65<=e['dec']<=2.51]
            es.sort(key=lambda e:-e['ev'])
            lines='; '.join(f"{MK[e['mk']]} {int(e['tgt'])}+ {e['am']} (φέτος {e['hc']}, πέρσι {e['hl']})" for e in es[:3]) or 'χωρίς γραμμή DraftKings στο εύρος 1.65–2.50'
            pk=PICK.get((t,name))
            if pk: dec,cls=pk,'y'
            elif es and es[0]['ev']>0.15:
                hc,nc=map(int,es[0]['hc'].split('/'))
                dec,cls=('όχι: φετινός ρόλος κάτω από τη γραμμή (πέρσι άλλος ρόλος/ομάδα)' if hc/nc<0.6 else 'όχι: πάγκος/λεπτά ή η γραμμή έχει κινηθεί (βλ. «Κοντά»)'),''
            elif es: dec,cls='χωρίς αξία','n'
            else: dec,cls='—',''
            rows.append(f"<tr><td>{t}</td><td>{html.escape(name)}</td><td>{len(v)}</td><td>{m('min'):.0f}</td><td>{m('pts'):.1f}</td><td>{m('reb'):.1f}</td><td>{m('ast'):.1f}</td><td>{m('tpm'):.1f}</td><td>{', '.join(str(x['pts']) for x in v[-5:])}</td><td>{lines}</td><td class=\"{cls}\">{dec}</td></tr>")
    out.append(f"<details><summary>{a} @ {h} · DraftKings: {o.get('details')} · O/U {o.get('overUnder')}</summary>\n<div class=\"tbl\"><table><tr><th>Ομάδα</th><th>Παίκτης</th><th>GP</th><th>Λεπτά</th><th>Π</th><th>Ρ</th><th>Α</th><th>3P</th><th>Πόντοι (τελ. 5)</th><th>Καλύτερες γραμμές DraftKings (hit rate)</th><th>Απόφαση</th></tr>\n"+'\n'.join(rows)+"</table></div></details>")
open(base+'/scan.html','w').write('\n'.join(out)); print(len(out),'games',nplayers,'players')
