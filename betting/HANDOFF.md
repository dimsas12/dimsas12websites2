# Μεταφορά σε τοπική συνεδρία (υπολογιστής του χρήστη, Ελλάδα)

## Γιατί τοπικά
Η cloud συνεδρία βγαίνει στο internet από ΗΠΑ. Οι ελληνικές εταιρίες (Stoiximan, Novibet, Betano, bwin, Sportingbet,
Betsson, Elabet, Winmasters, Interwetten, Betshop, Pamestoixima, Vistabet, bet365.gr) μπλοκάρουν τη σύνδεση από ΗΠΑ.
Από τον υπολογιστή του χρήστη (σύνδεση Ελλάδας) ανοίγουν κανονικά. Στόχος: props ΟΛΩΝ των παικτών με over ΚΑΙ under
για Euroleague, GBL και ελληνικό ποδόσφαιρο, από τις δημόσιες σελίδες των εταιριών (όπως τις βλέπει ο χρήστης στον browser).

## Τι υπάρχει στο repo
- `betting/PROGRAM.md` — όλο το καθημερινό πρόγραμμα (κανόνες, κριτήρια (α)–(ζ), πηγές, παράδοση). Ακολούθησέ το.
- `betting/tools/` — όλα τα scripts (στο PROGRAM.md αναφέρονται ως `scratchpad/...`). Κάθε script παίρνει φάκελο
  δεδομένων ως όρισμα· τρέξε τα από έναν φάκελο δεδομένων εκτός repo.
- `betting/log.md` — ιστορικό δελτίων και P/L. `betting/deltia.html` — η σελίδα «Δελτία Ημέρας».
- `betting/sharpe_analysis.html` — ανάλυση του Sharpe Locks (μαθήματα για αγορές όγκου).

## Πηγές αποδόσεων
- NBA / NFL / τένις / μεγάλα πρωταθλήματα / Champions-Europa-Conference: ΜΟΝΟ DraftKings
  (ESPN odds API πάροχος 100 για props, ESPN pickcenter, Action Network book 68, scoresandodds).
- Euroleague / Eurocup / GBL / ελληνικό ποδόσφαιρο:
  - BetRivers (Kambi, δημόσιο): `https://eu-offering-api.kambicdn.com/offering/v2018/rsiuspa/listView/basketball.json?lang=en_US&market=US`
    και `.../betoffer/event/<id>.json?lang=en_US&market=US` (props N+ για τους βασικούς, μόνο over).
  - Pinnacle guest (δημόσιο): `https://guest.api.arcadia.pinnacle.com/0.1/leagues/<382 Euroleague | 437 GBL>/matchups`.
  - BetExplorer: `betting/tools/bx_scan.py` (bet365/BetMGM/Stake για αγορές ομάδας).
  - ΝΕΟ τοπικά: μία ελληνική εταιρία με πλήρη props (όλοι οι παίκτες, over/under), από τις δημόσιες σελίδες της.
    Βρες ποια ανοίγει και πώς φορτώνει τα props, γράψε script `betting/tools/gr_props.py`, και πρόσθεσέ το στο PROGRAM.md.

## Κατάσταση στις 06/11/2026
- Σύνολο: 11-14, P/L −2.77%, ROI −16% (βλ. log.md). Με γραμμές DraftKings: 5-1.
- Το cloud προγραμματισμένο task «Ημερήσια δελτία στοιχήματος» τρέχει ακόμα στις 21:01 στο cloud session.
  Όταν η τοπική ρύθμιση δουλέψει, ο χρήστης αποφασίζει αν θα το σταματήσει.
