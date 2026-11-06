import json,urllib.request,gzip,sys
S=sys.argv[1]; UA={"User-Agent":"Mozilla/5.0","Accept-Encoding":"gzip"}
def get(u):
    b=urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=40).read()
    if b[:2]==b'\x1f\x8b': b=gzip.decompress(b)
    return json.loads(b)
sb=get("https://site.api.espn.com/apis/site/v2/sports/soccer/uefa.champions/scoreboard?dates=20261104")
DOM={}
out=[]
for e in sb['events']:
    d=get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/uefa.champions/summary?event={e['id']}")
    json.dump(d,open(f"{S}/u4/{e['id']}.json",'w'))
    c=d['header']['competitions'][0]; o=(d.get('pickcenter') or [{}])[0]
    teams=[(x['homeAway'],x['team']['displayName'],x['team']['id']) for x in c['competitors']]
    inj={t['team']['displayName']:[(i['athlete']['displayName'],i.get('status')) for i in t.get('injuries',[])] for t in d.get('injuries',[])}
    print('\n###',e['date'],e['name'],'|',o.get('provider',{}).get('name'),o.get('details'),'O/U',o.get('overUnder'),o.get('overOdds'),o.get('underOdds'),'ML',o.get('homeTeamOdds',{}).get('moneyLine'),o.get('drawOdds',{}).get('moneyLine'),o.get('awayTeamOdds',{}).get('moneyLine'))
    print('   keys',[k for k in d.keys()])
    for k,v in inj.items(): print('   inj',k,v)
    for ha,nm,tid in teams:
        res=[]
        for lg in ['uefa.champions','eng.1','esp.1','ita.1','ger.1','fra.1','tur.1','por.1','ned.1','aut.1','nor.1','ukr.1','svk.1','gre.1','cze.1','bel.1','aze.1','eng.league_cup','ita.coppa_italia','esp.copa_del_rey','ger.dfb_pokal','fra.coupe_de_france','uefa.europa']:
            try: s=get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{lg}/teams/{tid}/schedule?season=2026")
            except Exception: continue
            for g in s.get('events',[]):
                cc=g['competitions'][0]
                if not cc['status']['type'].get('completed'): continue
                sc={x['team']['id']:int(float(x['score']['value'])) for x in cc['competitors']}
                if tid not in sc: continue
                opp=[x['team']['displayName'] for x in cc['competitors'] if x['team']['id']!=tid][0]
                h=[x['homeAway'] for x in cc['competitors'] if x['team']['id']==tid][0]
                res.append((g['date'][:10],lg.split('.')[0][:4],h[0],opp[:14],sc[tid],[v for k,v in sc.items() if k!=tid][0]))
        res=sorted(set(res))[-10:]
        gf=sum(r[4] for r in res)/max(1,len(res)); ga=sum(r[5] for r in res)/max(1,len(res))
        print(f"   {ha:4s} {nm:22s} last{len(res)} GF {gf:.2f} GA {ga:.2f} |", '; '.join(f"{r[0][5:]} {r[1]} {r[2]} {r[3]} {r[4]}-{r[5]}" for r in res))
