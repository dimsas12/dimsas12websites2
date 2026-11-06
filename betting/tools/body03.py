head=open('head.html').read().rstrip(); head=head[:-len('<div class="wrap">')]
exec(open('body02.py').read().split("wig_tbl_gr=")[0].split("\n",1)[1])  # slip(), sec()

pg_gr='''<div class="tbl"><table><tr><th>Ημ/νία</th><th>Αντίπαλος</th><th>Λεπτά</th><th>Πόντοι</th><th>Ριμπ.</th><th>Ασίστ</th><th>Ρ+Α</th><th>8+;</th></tr>
  <tr><td>20/10</td><td>@ Ντιτρόιτ</td><td>31</td><td>38</td><td>5</td><td>3</td><td>8</td><td class="y">ναι</td></tr>
  <tr><td>23/10</td><td>Νικς</td><td>34</td><td>15</td><td>7</td><td>2</td><td>9</td><td class="y">ναι</td></tr>
  <tr><td>26/10</td><td>Μπουλς</td><td>25</td><td>16</td><td>5</td><td>2</td><td>7</td><td class="n">όχι</td></tr>
  <tr><td>27/10 (b2b)</td><td>Νετς</td><td>36</td><td>20</td><td>7</td><td>7</td><td>14</td><td class="y">ναι</td></tr>
  <tr><td>30/10</td><td>Μπουλς</td><td>27</td><td>18</td><td>6</td><td>8</td><td>14</td><td class="y">ναι</td></tr>
  <tr><td>01/11</td><td>@ Ορλάντο</td><td>33</td><td>17</td><td>6</td><td>8</td><td>14</td><td class="y">ναι</td></tr></table></div>'''
pg_en=(pg_gr.replace('Ημ/νία','Date').replace('Αντίπαλος','Opponent').replace('Λεπτά','Min').replace('Πόντοι','Pts')
       .replace('Ριμπ.','Reb').replace('Ασίστ','Ast').replace('Ρ+Α','R+A').replace('8+;','8+?').replace('ναι','yes').replace('όχι','no')
       .replace('@ Ντιτρόιτ','@ Detroit').replace('Νικς','Knicks').replace('Μπουλς','Bulls').replace('Νετς','Nets').replace('@ Ορλάντο','@ Orlando'))
chk_gr='''<div class="tbl"><table><tr><th>Κριτήριο</th><th>Έλεγχος</th><th></th></tr>
  <tr><td>(α) hit rate</td><td>5/6 πάνω από 7.5 (83%)· τα τελευταία 3 όλα 14</td><td class="y">✔</td></tr>
  <tr><td>(β) περιθώριο</td><td>μέσος όρος 11.0 έναντι 7.5 (+47%)</td><td class="y">✔</td></tr>
  <tr><td>(γ) ψυχολογία</td><td>Σέλτικς με 5 σερί νίκες, ο Τζορτζ σε ρόλο δημιουργού</td><td class="y">✔</td></tr>
  <tr><td>(δ) λεπτά/απουσίες</td><td>βασικός, 25–36 λεπτά, όχι στη λίστα τραυματιών· back-to-back (το προηγούμενο b2b έπαιξε 36 λεπτά)</td><td class="y">✔ (με σημείωση)</td></tr>
  <tr><td>(ε) αγορά</td><td>η DraftKings έχει την καλύτερη τιμή στο Over· καμία κίνηση κατά</td><td class="y">✔</td></tr>
  <tr><td>(στ) EV</td><td>+15%</td><td class="y">✔</td></tr>
  <tr><td>(ζ) θέση</td><td>ο νέος ρόλος δημιουργού δεν έχει περάσει στη γραμμή</td><td class="y">✔</td></tr></table></div>'''
chk_en='''<div class="tbl"><table><tr><th>Criterion</th><th>Check</th><th></th></tr>
  <tr><td>(α) hit rate</td><td>5/6 over 7.5 (83%); the last 3 were all 14</td><td class="y">✔</td></tr>
  <tr><td>(β) margin</td><td>average 11.0 vs 7.5 (+47%)</td><td class="y">✔</td></tr>
  <tr><td>(γ) psychology</td><td>Celtics on a 5-game win streak, George in a playmaking role</td><td class="y">✔</td></tr>
  <tr><td>(δ) minutes/absences</td><td>starter, 25–36 minutes, not on the injury list; back-to-back (he played 36 minutes in the previous one)</td><td class="y">✔ (with a note)</td></tr>
  <tr><td>(ε) market</td><td>DraftKings has the best Over price; no move against</td><td class="y">✔</td></tr>
  <tr><td>(στ) EV</td><td>+15%</td><td class="y">✔</td></tr>
  <tr><td>(ζ) thesis</td><td>the new playmaking role hasn't reached the line</td><td class="y">✔</td></tr></table></div>'''

s1=slip('NBA','03:30 ώρα Ελλάδας','Σέλτικς @ Ρόκετς','Πολ Τζορτζ Over 7.5 ριμπάουντ + ασίστ','−140 (1.71) · DraftKings','0,75%','~67% · EV +15%',
 'Οι Σέλτικς έχουν 5 σερί νίκες (τελευταία 108-97 στο Ορλάντο χθες) και παίζουν το δεύτερο ματς back-to-back. Οι Ρόκετς επίσης (119-98 επί των Θάντερ χθες). Στους Ρόκετς λείπει ο Στίβεν Άνταμς, και ο Καπέλα είναι αμφίβολος.',
 '<b>Γιατί κερδίζει:</b> Η θέση: ο Τζορτζ έχει γίνει ο δεύτερος δημιουργός των Σέλτικς (7, 8, 8 ασίστ στα 3 τελευταία, 14 Ρ+Α σε καθένα), αλλά η DraftKings κρατά τη γραμμή στο 7.5, σαν να είναι ακόμα ο σκόρερ των πρώτων αγώνων (2–3 ασίστ). Πέρασε το 7.5 σε 5 από τα 6 ματς, με μέσο όρο 11.0. Στο προηγούμενο back-to-back έπαιξε 36 λεπτά και έκανε 14.',
 sec([('Η θέση: γιατί η DraftKings κάνει λάθος','<p>Στα 3 πρώτα ματς ο Τζορτζ είχε 2–3 ασίστ. Στα 3 τελευταία είχε 7, 8 και 8, γιατί οι Σέλτικς περνούν πια πολλές επιθέσεις από τα χέρια του (ο Ντέρικ Γουάιτ πέφτει: 4 και 2 ασίστ στα 2 τελευταία). Τα ριμπάουντ του είναι σταθερά 5–7. Η γραμμή 7.5 αντιστοιχεί στον παλιό ρόλο. Με τον νέο ρόλο, ακόμα και με 5 ριμπάουντ, θέλει μόνο 3 ασίστ.</p>'),
      ('Η γραμμή και η αγορά (DraftKings)','<p>DraftKings (scoresandodds, 21:05): Ρ+Α Over 7.5 <b>−140 (1.71)</b>, η καλύτερη τιμή της αγοράς. Ματς: Ρόκετς −5.5, σύνολο 216.5 (ESPN, πάροχος DraftKings).</p>'),
      ('Στατιστικά (ESPN game logs)',pg_gr+'<p>Μέσος όρος 29 λεπτά, 5.8 ριμπάουντ, 5.2 ασίστ.</p>'),
      ('H2H','<p>Δεν παίζει ρόλο για αυτό το prop. Δεν έχω αξιόπιστα φετινά H2H.</p>'),
      ('Απουσίες και ρόλοι','<ul><li>Ρόκετς: Στίβεν Άνταμς εκτός, Καπέλα και Σέπαρντ αμφίβολοι. Λιγότερο μέγεθος στο ριμπάουντ.</li><li>Σέλτικς: Μίτσελ Ρόμπινσον αμφίβολος (ξεκούραση), Ντέβιν Κάρτερ εκτός. Αν λείψει ο Ρόμπινσον, μένουν περισσότερα ριμπάουντ για τα φτερά.</li></ul>'),
      ('Ψυχολογία και πλαίσιο','<ul><li>Οι Σέλτικς έχουν 5 σερί νίκες, με τον Τζορτζ σταθερό σε κάθε ματς. Δεν υπάρχει λόγος να αλλάξει ο ρόλος του.</li><li>Ματς κορυφής (Ρόκετς 6-1 στις τελευταίες 7): ανταγωνιστικό, όχι περίπατος, άρα πλήρη λεπτά.</li></ul>'),
      ('Το μοντέλο','<p>Συρρικνωμένος μέσος (6 ματς, βάρος 50%) 9.3, και P(Over) ≈ 67%. Αν κρατήσω μόνο τα 3 τελευταία, βγαίνει πάνω από 80%, αλλά δεν το χρησιμοποιώ, για ασφάλεια.</p>'),
      ('Έλεγχος κριτηρίων (α)–(ζ)',chk_gr),
      ('Κατά / ρίσκα','<ul><li>Back-to-back: μπορεί να παίξει λιγότερο (στο ματς με 25 λεπτά έκανε 7).</li><li>Σε ήσυχο βράδυ στις ασίστ (2–3) χρειάζεται 6 ριμπάουντ.</li></ul>'),
      ('Συμπέρασμα','<p><span class="ev">EV = 0.67 × 1.71 − 1 = +15%</span>. Ποντάρισμα <b>0,75%</b> (το P/L είναι ακόμα αρνητικό). Ελάχιστη απόδοση: <b>−154 (1.65)</b>.</p>')]),
 sec([('The thesis: why DraftKings is wrong','<p>In his first 3 games George had 2–3 assists. In the last 3 he had 7, 8 and 8, because Boston now runs many possessions through him (Derrick White is fading: 4 and 2 assists in the last 2). His rebounds are steady at 5–7. The 7.5 line matches the old role. In the new role, even with 5 rebounds, he only needs 3 assists.</p>'),
      ('The line and the market (DraftKings)','<p>DraftKings (scoresandodds, 21:05): R+A Over 7.5 <b>−140 (1.71)</b>, the best price in the market. Game: Rockets −5.5, total 216.5 (ESPN, DraftKings provider).</p>'),
      ('Stats (ESPN game logs)',pg_en+'<p>Average 29 minutes, 5.8 rebounds, 5.2 assists.</p>'),
      ('H2H','<p>It doesn\'t matter for this prop. I have no reliable H2H from this season.</p>'),
      ('Absences and roles','<ul><li>Rockets: Steven Adams out, Capela and Sheppard questionable. Less size on the boards.</li><li>Celtics: Mitchell Robinson questionable (rest), Devin Carter out. If Robinson sits, more rebounds are left for the wings.</li></ul>'),
      ('Psychology and context','<ul><li>Boston have won 5 straight, with George steady in every game. There is no reason for his role to change.</li><li>A top game (Rockets 6-1 in their last 7): competitive, not a blowout, so full minutes.</li></ul>'),
      ('The model','<p>Shrunk mean (6 games, 50% weight) 9.3, and P(Over) ≈ 67%. Using only the last 3 games it comes out above 80%, but I don\'t use that, to stay safe.</p>'),
      ('Criteria check (α)–(ζ)',chk_en),
      ('Against / risks','<ul><li>Back-to-back: he may play less (in the 25-minute game he had 7).</li><li>On a quiet assist night (2–3) he needs 6 rebounds.</li></ul>'),
      ('Conclusion','<p><span class="ev">EV = 0.67 × 1.71 − 1 = +15%</span>. Stake <b>0.75%</b> (total P/L is still negative). Minimum price: <b>−154 (1.65)</b>.</p>')]))

top=f'''<div class="wrap">
<header>
  <h1>Δελτίο Τρίτης 3/11</h1>
  <p>Νέος κανόνας: κάθε δελτίο ξεκινά από μια <b>αναλυτική θέση</b>, δηλαδή γιατί η γραμμή της DraftKings είναι λάθος (ρόλος, απουσίες, κίνητρο, αριθμοί). Χωρίς θέση δεν υπάρχει δελτίο. Σήμερα πέρασε τα 7 κριτήρια μόνο <b>ένα</b>.</p>
</header>
<div class="total"><span>Δελτία <b>1</b></span><span>Σύνολο ρίσκου <b>0,75%</b> bankroll</span><span>Ενημέρωση <b>02/11/2026 · 21:10</b></span></div>
{s1}
<section class="slip side">
  <h2>Τι σάρωσα</h2>
  <p class="note"><b>NBA:</b> 15 ματς (02:00–05:00), 325 παίκτες rotation. Από τις 3.041 γραμμές props του scoresandodds, οι <b>530</b> ήταν της DraftKings. 34 πλευρές βασικών παικτών πέρασαν τα κριτήρια (α)–(β), και οι περισσότερες ήταν under, που απορρίφθηκαν λόγω κανόνα ή επειδή ο ρόλος μεγαλώνει από απουσίες. Αναλυτική θέση και όλα τα κριτήρια πέρασε μόνο ο Τζορτζ. <b>Ποδόσφαιρο:</b> 9 ματς Champions League με γραμμές DraftKings και 2 του Κυπέλλου Ισπανίας χωρίς γραμμές. <b>Euroleague:</b> χωρίς γραμμές DraftKings στις πηγές.</p>
  <p class="note">Περιορισμός: το scoresandodds δείχνει τη γραμμή της DraftKings μόνο όταν έχει την καλύτερη τιμή. Γι' αυτό πολλοί παίκτες εμφανίζονται «χωρίς γραμμή DraftKings».</p>
  <h3>NBA</h3>
'''
bot=f'''
  <h3>Ποδόσφαιρο: Champions League (DraftKings μέσω ESPN)</h3>
  <div class="tbl"><table><tr><th>Αγώνας</th><th>Γκολ υπέρ–κατά (γηπ./φιλ.)</th><th>Γραμμές DraftKings (δεκαδικά)</th><th>EV (blend)</th><th>Απόφαση</th></tr>
{open('fb03.html').read()}
  </table></div>
</section>
<section class="slip side">
  <h2>Κοντά αλλά όχι</h2>
  <ul>
    <li><b>Τζαμπάρι Σμιθ Π+Α O16.5 (−113):</b> υπάρχει θέση (λείπει ο Άνταμς, παίζει 33–39 λεπτά, 22 και 30 πόντοι στα 2 τελευταία). Όμως πέρασε 5/7, με δύο καθαρές αποτυχίες (8, 16), και είναι back-to-back. Κράτησα 1 δελτίο.</li>
    <li><b>Φοξ O5.5 ασίστ (−139):</b> 5/6, αλλά οι Σπερς είναι −16.5. Σε περίπατο κόβονται λεπτά (κριτήριο δ).</li>
    <li><b>Λαμέλο, Μπρίτζες και Τζέις Ρίτσαρντσον under:</b> EV +25–30%, αλλά είναι under, και οι ρόλοι μεγαλώνουν από απουσίες (Έντουαρντς, Μπανκέρο, Σαγκς). Το μάθημα της 30/10.</li>
    <li><b>Βιγιαρεάλ–Παρί Over 3.5 (−115):</b> τα 6 ματς των δύο στο Champions League είχαν 4+ γκολ, αλλά στο πρωτάθλημα μόνο 4/8 (Παρί) και 5/8 (Βιγιαρεάλ). Το δείγμα είναι μικρό για θέση.</li>
    <li><b>Σάι Ρ O3.5 (−141):</b> 4/6, χωρίς αξία.</li>
  </ul>
</section>
<section class="slip side">
  <h2>Απολογισμός</h2>
  <div class="tbl"><table><tr><th>Αγώνας</th><th>Επιλογή</th><th>Stake</th><th>Αποτέλεσμα</th><th>P/L</th></tr>
  <tr><td>27/10 – 1/11 (19 στοιχήματα)</td><td>βλ. betting/log.md</td><td>13.5%</td><td>6-13</td><td class="n">−5.86%</td></tr>
  <tr><td>Ράπτορς @ Λέικερς</td><td>Λούκα O4.5 τρίποντα (2.16)</td><td>0.75%</td><td class="y">8 τρίποντα, 45 πόντοι ✅</td><td class="y">+0.87%</td></tr>
  <tr><td>Σασουόλο – Φιορεντίνα</td><td>Over 3.5 (2.35)</td><td>0.5%</td><td class="y">2-2 ✅</td><td class="y">+0.68%</td></tr>
  <tr><td>Λα Κορούνια – Οσασούνα</td><td>Λα Κορούνια (2.15)</td><td>0.75%</td><td>εκκρεμεί (22:00)</td><td>–</td></tr></table></div>
  <p class="note">Κριθέντα: 8-13, <b>−4.31%</b> σε 14.75% ποντάρισμα (ROI −29%). Δύο στα δύο τα δελτία της DraftKings που κρίθηκαν μέχρι τώρα.</p>
</section>
</div>'''
page=head+top+open('scan_dk03.html').read()+bot
open('/home/user/dimsas12websites2/betting/deltia.html','w').write(page)
print(len(page),page.count('<details>'),page.count('</details>'))
