"""scoresandodds NBA props: keep the sides whose displayed best price is DraftKings; write dk_props.json."""
import re, html, json, sys, urllib.request
MARKETS = {"pts": "points", "reb": "rebounds", "ast": "assists", "tpm": "3-pointers",
           "pra": "points,-rebounds,-&-assists", "pr": "points-&-rebounds", "pa": "points-&-assists", "ra": "rebounds-&-assists"}
def fetch(slug):
    r = urllib.request.Request(f"https://www.scoresandodds.com/nba/props/{slug}", headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124"})
    return urllib.request.urlopen(r, timeout=40).read().decode("utf-8", "ignore")
def dec(a):
    a = int(a); return round(1 + (a / 100 if a > 0 else 100 / -a), 3)
out = []
for key, slug in MARKETS.items():
    try: raw = fetch(slug)
    except Exception as e: print(key, 'ERR', e); continue
    for row in raw.split('<li class="border"')[1:]:
        name = re.search(r'data-name="([^"]+)"', row)
        game = re.search(r'<span class="bold small gray">([^<]+)</span>', row)
        pname = re.search(r'<a href="/prop-bets/\d+/[^"]+">([^<]+)</a>', row)
        sides = re.findall(r'<span class="data-moneyline">([ou])([\d.]+)</span>\s*<small class="data-odds[^"]*">([+-]?\d+)</small>.*?alt="([a-z0-9]+)"', row, re.S)
        for side, line, odds, book in sides:
            out.append(dict(market=key, player=html.unescape(pname.group(1)) if pname else name.group(1), game=game.group(1) if game else '', side=side, line=float(line), american=int(odds), dec=dec(odds), book=book))
json.dump(out, open('dk/sao_all.json', 'w'), indent=0)
dk = [o for o in out if o['book'] == 'draftkings']
json.dump(dk, open('dk/dk_props.json', 'w'), indent=0)
from collections import Counter
print(len(out), 'lines;', len(dk), 'with DraftKings as displayed book;', Counter(o['book'] for o in out))
print(Counter(o['game'] for o in dk))
