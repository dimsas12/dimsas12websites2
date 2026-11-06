import openpyxl,glob,sys,json,re
S=sys.argv[1]; rows=[];streak=[]
for f in sorted(glob.glob(S+'/sharpe_in/*.xlsx')):
    wb=openpyxl.load_workbook(f,data_only=True)
    for ws in wb.worksheets:
        for r in ws.iter_rows(min_row=3,values_only=True):
            if r[0] is not None and r[2]:
                o=r[4]; o=float(str(o).replace('+','')) if o not in (None,'') else None
                rows.append(dict(date=str(r[0])[:10],sport=str(r[1]).strip(),who=str(r[2]).strip(),prop=str(r[3]).strip(),odds=o,res=str(r[5]).strip(),risk=r[6],won=r[7]))
            if r[10]: streak.append(dict(sheet=ws.title,player=r[10],prop=r[11],leg=r[12],res=r[13]))
        # other columns beyond 14?
        extra=[r[14:] for r in ws.iter_rows(min_row=1,values_only=True) if any(c is not None for c in r[14:])]
        print(ws.title,'extra cols rows',len(extra), extra[:5])
json.dump(dict(rows=rows,streak=streak),open(S+'/sharpe.json','w'))
print(len(rows),len(streak))
