import json,glob,sys,os
S=sys.argv[1]; out={}; games={}
for f in glob.glob(S+'/shp/sum/*.json'):
    d=json.load(open(f)); c=d['header']['competitions'][0]
    if not c['status']['type'].get('completed'): continue
    date=c['date']; gid=os.path.basename(f)[:-5]
    sc={x['team']['abbreviation']:int(x.get('score') or 0) for x in c['competitors']}
    for side in d.get('boxscore',{}).get('players',[]):
        ab=side['team']['abbreviation']; opp=[k for k in sc if k!=ab][0]
        st=side['statistics'][0]; keys=st['keys']
        dnp=[]
        for a in st['athletes']:
            nm=a['athlete']['displayName']; aid=a['athlete']['id']
            if a.get('didNotPlay') or not a.get('stats'):
                dnp.append(nm); continue
            v=dict(zip(keys,a['stats']))
            def num(k):
                try: return float(v.get(k,'0').split('-')[0])
                except: return 0
            tp=v.get('threePointFieldGoalsMade-threePointFieldGoalsAttempted','0-0').split('-')
            out.setdefault(aid,{'name':nm,'g':[]})['g'].append(dict(date=date,gid=gid,team=ab,opp=opp,min=num('minutes'),pts=num('points'),reb=num('rebounds'),ast=num('assists'),tpm=float(tp[0]),tpa=float(tp[1]) if len(tp)>1 else 0,starter=a.get('starter',False),margin=sc[ab]-sc[opp]))
        games.setdefault(gid,{})[ab]=dict(date=date,dnp=dnp,opp=opp)
for a in out.values(): a['g'].sort(key=lambda g:g['date'])
json.dump(dict(players=out,games=games),open(S+'/shp/box.json','w'))
print(len(out),'players',len(games),'games')
