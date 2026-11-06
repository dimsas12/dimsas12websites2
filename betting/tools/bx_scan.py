"""BetExplorer: fixtures of a league page + odds (1X2/HA, O/U, AH) per bookmaker. Usage: bx_scan.py OUTDIR league_path [days_ahead]"""
import re,json,sys,html,urllib.request,gzip,os,datetime
OUT=sys.argv[1]; LP=sys.argv[2].strip('/'); DAYS=int(sys.argv[3]) if len(sys.argv)>3 else 1
UA={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36","Accept-Encoding":"gzip"}
def get(u,ref=None):
    h=dict(UA); 
    if ref: h.update({"Referer":ref,"X-Requested-With":"XMLHttpRequest"})
    b=urllib.request.urlopen(urllib.request.Request(u,headers=h),timeout=40).read()
    if b[:2]==b'\x1f\x8b': b=gzip.decompress(b)
    return b.decode('utf-8','ignore')
os.makedirs(OUT,exist_ok=True)
base="https://www.betexplorer.com/"+LP+"/"
s=get(base); s=s[s.find("leaguefixtures"):]; s=s[:s.find("Show all fixtures")]
links=[]
for m in re.finditer(r'<tr[^>]*>(.*?)</tr>',s,re.S):
    r=m.group(1); a=re.search(r'href="(/'+re.escape(LP)+r'/([^/"]+)/([A-Za-z0-9]{8})/)"',r)
    if not a: continue
    txt=re.sub(r'\s+',' ',re.sub(r'<[^>]+>',' ',r))
    dt=re.search(r'(Today \d\d:\d\d|Tomorrow \d\d:\d\d|\d\d\.\d\d\. \d\d:\d\d)',txt)
    links.append((a.group(1),a.group(2),a.group(3),dt.group(1) if dt else ''))
seen=set(); res=[]
today=datetime.date.today()
for href,slug,mid,dt in links:
    if mid in seen: continue
    seen.add(mid)
    # date filter: dt like "06.11. 20:15" or "Today 20:15"
    ok=('Today' in dt) or ('Tomorrow' in dt and DAYS>=1)
    mm=re.match(r'(\d\d)\.(\d\d)\.',dt)
    if mm:
        d=datetime.date(today.year,int(mm.group(2)),int(mm.group(1)))
        ok=0<=(d-today).days<=DAYS
    if not ok: continue
    ref="https://www.betexplorer.com"+href; g=dict(slug=slug,id=mid,dt=dt,url=ref,odds={})
    for bt in ('1x2','ha','ou','ah','bts'):
        try: d=json.loads(get(f"https://www.betexplorer.com/match-odds/{mid}/1/{bt}/odds/?lang=en",ref))
        except Exception: continue
        rows=[]
        hdr=None
        for r in re.findall(r'<tr[^>]*>(.*?)</tr>',d.get('odds',''),re.S):
            t=re.sub(r'\s+',' ',html.unescape(re.sub(r'<[^>]+>',' ',r))).strip()
            o=[float(x) for x in re.findall(r'data-odd="([\d.]+)"',r)]
            if not o: continue
            book=t.split(' ')[0]; line=None
            mm2=re.match(r'\S+ ([+-]?[\d.,]+) ',t+' ')
            if bt in ('ou','ah') and mm2: line=float(mm2.group(1).replace(',','.').rstrip('.'))
            rows.append(dict(book=book,line=line,odds=o))
        if rows: g['odds'][bt]=rows
    res.append(g)
json.dump(res,open(f"{OUT}/{LP.replace('/','_')}.json",'w'),indent=0)
for g in res:
    print('##',g['dt'],g['slug'])
    for bt,rows in g['odds'].items():
        if bt in ('1x2','ha','bts'): print('  ',bt,'; '.join(f"{r['book']} {r['odds']}" for r in rows))
        else:
            # main line = closest to even odds
            mains=sorted(rows,key=lambda r:abs(r['odds'][0]-r['odds'][-1]))[:3]
            print('  ',bt,'; '.join(f"{r['book']} {r['line']} {r['odds']}" for r in mains))
