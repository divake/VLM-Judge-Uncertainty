#!/usr/bin/env python3
"""Exploratory: Cohen's kappa and conformal interval width (Capital One reviewer ask).

Four analyses, all re-analysis of existing results (no inference, no CP refit):
  A. Judge-level kappa (unweighted/linear/quadratic) vs Pearson, both benchmarks
  B. Kappa stratified by conformal interval-width decile
  C. Does width predict |judge - human|?
  D. Task-level kappa vs task-level width across the 14 MLLM-Judge categories

Findings are recorded in paper/Paper_2026_AAAI_VLM_JUDGE/CHECKLIST.md section 7.
Run from repo root:  conda run -n env_py311 python scripts/kappa_width_analysis.py
"""


# ======================================================================
# A. Judge-level kappa vs Pearson
# ======================================================================
import csv, numpy as np
from collections import Counter

def kappas(y, p):
    y=np.asarray(y); p=np.asarray(p); n=len(y); K=5
    O=np.zeros((K,K))
    for a,b in zip(y,p): O[a-1,b-1]+=1
    O/=n
    ry=O.sum(1); rp=O.sum(0); E=np.outer(ry,rp)
    out={}
    for name,wf in [('unweighted',lambda i,j: 0.0 if i==j else 1.0),
                    ('linear',    lambda i,j: abs(i-j)/(K-1)),
                    ('quadratic', lambda i,j: (i-j)**2/(K-1)**2)]:
        W=np.array([[wf(i,j) for j in range(K)] for i in range(K)])
        out[name]=1-(W*O).sum()/(W*E).sum()
    out['p_o']=np.trace(O)
    out['p_e']=np.trace(E)
    return out

print("=== MLLM-as-a-Judge: judge vs human ===")
print(f"{'judge':<14}{'kappa_unw':>10}{'kappa_lin':>10}{'kappa_quad':>11}{'p_o':>8}{'pearson':>9}")
for j,f in [('LLaVA-Critic','results/v2/features_s2.csv'),
            ('Phi-4','results/v2_phi4/features_s2.csv'),
            ('Gemini','results/v2_gemini/features_s2.csv')]:
    r=[x for x in csv.DictReader(open(f)) if x['parsed_score'] not in ('','nan')]
    y=[int(float(x['gt_score'])) for x in r]; p=[int(float(x['parsed_score'])) for x in r]
    k=kappas(y,p); pe=np.corrcoef(y,p)[0,1]
    print(f"{j:<14}{k['unweighted']:>10.3f}{k['linear']:>10.3f}{k['quadratic']:>11.3f}{k['p_o']:>8.3f}{pe:>9.3f}")

print("\n=== Polaris: judge vs human (LLaVA-Critic) ===")
r=[x for x in csv.DictReader(open('results/v2_polaris/features_s2.csv')) if x['parsed_score'] not in ('','nan')]
y=[int(float(x['gt_score'])) for x in r]; p=[int(float(x['parsed_score'])) for x in r]
k=kappas(y,p)
print(f"{'LLaVA-Critic':<14}{k['unweighted']:>10.3f}{k['linear']:>10.3f}{k['quadratic']:>11.3f}{k['p_o']:>8.3f}{np.corrcoef(y,p)[0,1]:>9.3f}")


# ======================================================================
# B. Kappa by interval-width decile
# ======================================================================
import csv, numpy as np
from collections import defaultdict

def kappa_unw(y,p,K=5):
    y=np.asarray(y);p=np.asarray(p);n=len(y)
    O=np.zeros((K,K))
    for a,b in zip(y,p):O[a-1,b-1]+=1
    O/=n; E=np.outer(O.sum(1),O.sum(0))
    po=np.trace(O);pe=np.trace(E)
    return (po-pe)/(1-pe), po, n

rows=defaultdict(list)
for r in csv.DictReader(open('results/rebuttal/exp4/per_instance.csv')):
    try:
        rows[r['judge']].append((float(r['width_raw']), int(float(r['gt_score'])), int(float(r['parsed_score']))))
    except (ValueError, KeyError):
        pass

print("Unweighted Cohen's kappa by RAW conformal interval-width decile (MLLM-as-a-Judge)")
print("D1 = narrowest intervals, D10 = widest.  Pooled over 10 seeds.\n")
for judge in ['LLaVA-Critic','Phi-4','Gemini']:
    d=sorted(rows[judge]); w=np.array([x[0] for x in d])
    qs=np.quantile(w,np.linspace(0,1,11))
    line_k=[];line_a=[]
    for i in range(10):
        lo,hi=qs[i],qs[i+1]
        sel=[x for x in d if (lo<=x[0]<=hi if i==9 else lo<=x[0]<hi)]
        if len(sel)<50: line_k.append('  n/a'); line_a.append('  n/a'); continue
        k,po,n=kappa_unw([x[1] for x in sel],[x[2] for x in sel])
        line_k.append(f"{k:5.2f}"); line_a.append(f"{po:5.2f}")
    allk,_,ntot=kappa_unw([x[1] for x in d],[x[2] for x in d])
    print(f"{judge}  (overall kappa={allk:.3f}, width range {w.min():.2f}-{w.max():.2f}, n={ntot})")
    print("   kappa: "+" ".join(line_k))
    print("   exact: "+" ".join(line_a))
    # correlation between width and per-instance correctness
    corr=np.corrcoef([x[0] for x in d],[1 if x[1]==x[2] else 0 for x in d])[0,1]
    print(f"   Pearson(width, exact-match indicator) = {corr:+.3f}\n")


# ======================================================================
# C. Width vs |judge - human|
# ======================================================================
import csv, numpy as np
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr

rows=defaultdict(list)
for r in csv.DictReader(open('results/rebuttal/exp4/per_instance.csv')):
    try:
        rows[r['judge']].append((float(r['width_raw']), float(r['width_adj']),
                                 int(float(r['gt_score'])), int(float(r['parsed_score']))))
    except (ValueError,KeyError): pass

print("Q: does conformal WIDTH predict |judge - human| ?   (MLLM-as-a-Judge, 10 seeds pooled)\n")
print(f"{'judge':<14}{'n':>7}{'Spearman(w,|err|)':>20}{'p':>10}{'Pearson':>10}")
for j in ['LLaVA-Critic','Phi-4','Gemini']:
    d=rows[j]; w=np.array([x[0] for x in d]); e=np.array([abs(x[2]-x[3]) for x in d])
    rs,ps=spearmanr(w,e); rp,_=pearsonr(w,e)
    print(f"{j:<14}{len(d):>7}{rs:>20.3f}{ps:>10.1e}{rp:>10.3f}")

print("\nMean |judge - human| by width decile (D1 narrowest -> D10 widest):")
for j in ['LLaVA-Critic','Phi-4','Gemini']:
    d=rows[j]; w=np.array([x[0] for x in d]); e=np.array([abs(x[2]-x[3]) for x in d])
    qs=np.quantile(w,np.linspace(0,1,11)); out=[]
    for i in range(10):
        m=(w>=qs[i])&(w<=qs[i+1]) if i==9 else (w>=qs[i])&(w<qs[i+1])
        out.append(f"{e[m].mean():4.2f}" if m.sum()>50 else " n/a")
    print(f"  {j:<14}"+" ".join(out)+f"   (overall MAE {e.mean():.2f})")

print("\nFraction with |err|>=2 ('badly wrong') by width decile:")
for j in ['LLaVA-Critic','Phi-4','Gemini']:
    d=rows[j]; w=np.array([x[0] for x in d]); e=np.array([abs(x[2]-x[3]) for x in d])
    qs=np.quantile(w,np.linspace(0,1,11)); out=[]
    for i in range(10):
        m=(w>=qs[i])&(w<=qs[i+1]) if i==9 else (w>=qs[i])&(w<qs[i+1])
        out.append(f"{(e[m]>=2).mean():4.2f}" if m.sum()>50 else " n/a")
    print(f"  {j:<14}"+" ".join(out)+f"   (overall {(e>=2).mean():.2f})")


# ======================================================================
# D. Task-level kappa vs width
# ======================================================================
import csv, numpy as np
from collections import defaultdict
from scipy.stats import spearmanr

# sample_id -> dataset (same ordering across judges; verify via gt_score)
m={}; gtref={}
for r in csv.DictReader(open('results/v2/error_analysis.csv')):
    m[r['sample_id']]=r['orig_dataset']; gtref[r['sample_id']]=int(float(r['gt_score']))

def kap(y,p,mode,K=5):
    y=np.asarray(y);p=np.asarray(p);n=len(y)
    O=np.zeros((K,K))
    for a,b in zip(y,p):O[a-1,b-1]+=1
    O/=n;E=np.outer(O.sum(1),O.sum(0))
    if mode=='unw': W=1-np.eye(K)
    else: W=np.array([[abs(i-j)/(K-1) for j in range(K)] for i in range(K)])
    return 1-(W*O).sum()/(W*E).sum()

JUDGES={'LLaVA-Critic':('results/v2/features_s2.csv','results/v2/r2ccp_per_dataset.csv'),
        'Phi-4':('results/v2_phi4/features_s2.csv','results/v2_phi4/r2ccp_per_dataset.csv'),
        'Gemini':('results/v2_gemini/features_s2.csv','results/v2_gemini/r2ccp_per_dataset.csv')}

for judge,(feat,perds) in JUDGES.items():
    try: W={r['dataset']:(float(r['width_raw']),float(r['pearson'])) for r in csv.DictReader(open(perds))}
    except FileNotFoundError:
        print(f"\n{judge}: no per-dataset R2CCP file, skipping"); continue
    g=defaultdict(list); mism=0
    for r in csv.DictReader(open(feat)):
        sid=r['sample_id']
        if sid not in m or r['parsed_score'] in ('','nan'): continue
        if int(float(r['gt_score']))!=gtref[sid]: mism+=1
        g[m[sid]].append((int(float(r['gt_score'])),int(float(r['parsed_score']))))
    rows=[]
    for d,v in g.items():
        if d not in W or len(v)<100: continue
        y=[x[0] for x in v]; p=[x[1] for x in v]
        rows.append((d,len(v),W[d][0],W[d][1],kap(y,p,'unw'),kap(y,p,'lin')))
    if not rows: continue
    rows.sort(key=lambda x:-x[4])
    print(f"\n===== {judge}  (gt mismatches: {mism}) =====")
    print(f"{'task':<20}{'n':>5}{'width':>8}{'pearson':>9}{'kappa_unw':>11}{'kappa_lin':>11}")
    for d,n,w,pe,ku,kl in rows: print(f"{d:<20}{n:>5}{w:>8.2f}{pe:>9.3f}{ku:>11.3f}{kl:>11.3f}")
    wd=[x[2] for x in rows]; pe=[x[3] for x in rows]; ku=[x[4] for x in rows]; kl=[x[5] for x in rows]
    print(f"  across {len(rows)} tasks:")
    for nm,v in [('Pearson',pe),('kappa_unw',ku),('kappa_lin',kl)]:
        rs,ps=spearmanr(v,wd); print(f"    Spearman({nm:<10}, width) = {rs:+.3f}   p={ps:.3f}")
