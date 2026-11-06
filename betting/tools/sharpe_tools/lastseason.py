import json,sys,os,unicodedata,urllib.request,gzip,re
S=sys.argv[1]; B=json.load(open(S+'/shp/box.json')); P=B['players']
R=[r for r in json.load(open(S+'/sharpe.json'))['rows'] if r['sport']=='NBA']
def norm(s): return re.sub(r'[^a-z ]','',unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode().lower()).replace(' iii','').replace(' jr','').strip()
idx={norm(v['name']):k for k,v in P.items()}
alias={'robert willams':'robert williams'}
UA={"User-Agent":"Mozilla/5.0","Accept-Encoding":"gzip"}
def get(u):
    b=urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=40).read()
    if b[:2]==b'\x1f\x8b': b=gzip.decompress(b)
    return json.loads(b)
os.makedirs(S+'/shp/gl',exist_ok=True); miss=[]
for r in R:
    n=norm(r['who']); n=alias.get(n,n)
    aid=idx.get(n)
    if not aid: miss.append(r['who']); continue
    p=f"{S}/shp/gl/{aid}.json"
    if os.path.exists(p): continue
    try: d=get(f"https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{aid}/gamelog?season=2026")
    except Exception as e: d={'err':str(e)}
    json.dump(d,open(p,'w'))
print('unmatched',sorted(set(miss)))
