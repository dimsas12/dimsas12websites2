import json,glob,sys,os
S=sys.argv[1]; P={}; G={}
def num(x):
    try: return float(str(x).split('/')[0].split('-')[0])
    except: return 0.0
for f in glob.glob(S+'/shp/nfl/*.json'):
    d=json.load(open(f)); c=d['header']['competitions'][0]; gid=os.path.basename(f)[:-5]
    teams={x['team']['abbreviation']:x for x in c['competitors']}
    sc={k:int(v.get('score') or 0) for k,v in teams.items()}
    pc=[p for p in d.get('pickcenter',[]) if 'raft' in p.get('provider',{}).get('name','')] or d.get('pickcenter',[])
    pc=pc[0] if pc else {}
    home=[k for k,v in teams.items() if v['homeAway']=='home'][0]
    spread=pc.get('spread'); total=pc.get('overUnder')
    G[gid]=dict(date=c['date'],home=home,teams=list(teams),spread_home=spread,total=total,score=sc,prov=pc.get('provider',{}).get('name'))
    for side in d.get('boxscore',{}).get('players',[]):
        ab=side['team']['abbreviation']; opp=[k for k in teams if k!=ab][0]
        for cat in side['statistics']:
            keys=cat['keys']
            for a in cat['athletes']:
                aid=a['athlete']['id']; v=dict(zip(keys,a['stats']))
                rec=P.setdefault(aid,{'name':a['athlete']['displayName'],'pos':a['athlete'].get('position',{}).get('abbreviation'),'g':{}})
                g=rec['g'].setdefault(gid,dict(gid=gid,date=c['date'],team=ab,opp=opp,home=(ab==home)))
                if cat['name']=='passing':
                    ca=v.get('completions/passingAttempts','0/0').split('/'); g.update(cmp=num(ca[0]),patt=num(ca[1]) if len(ca)>1 else 0,pyds=num(v.get('passingYards')))
                if cat['name']=='rushing': g.update(ratt=num(v.get('rushingAttempts')),ryds=num(v.get('rushingYards')),rlong=num(v.get('longRushing')),rtd=num(v.get('rushingTouchdowns')))
                if cat['name']=='receiving': g.update(rec=num(v.get('receptions')),yds=num(v.get('receivingYards')),long=num(v.get('longReception')),tgt=num(v.get('receivingTargets')),ctd=num(v.get('receivingTouchdowns')))
for v in P.values(): v['g']=sorted(v['g'].values(),key=lambda g:g['date'])
json.dump(dict(players=P,games=G),open(S+'/shp/nfl_box.json','w'))
print(len(P),'players',len(G),'games', sum(1 for g in G.values() if g['prov']))
