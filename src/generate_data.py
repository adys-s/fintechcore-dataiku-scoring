# -*- coding: utf-8 -*-
# =====================================================================
# Génération des données synthétiques du projet FintechCore
# Produit : data/clients_db.csv (1 200 clients)
#           data/transactions_csv.csv (~10 800 échéances, séparateur ;)
# Les données sont fictives et reproductibles (graine aléatoire fixe).
# Usage : python src/generate_data.py   (depuis la racine du dépôt)
# =====================================================================
import numpy as np, pandas as pd
rng=np.random.default_rng(42)
N=1200
ids=[f"C{str(i).zfill(5)}" for i in range(1,N+1)]
age=rng.integers(19,75,N).astype(float)
revenu=np.round(rng.lognormal(10.35,0.45,N),-2)
contrat=rng.choice(["CDI","CDD","Independant","Interim","Retraite"],N,p=[.5,.15,.13,.1,.12])
region=rng.choice(["Auvergne-Rhone-Alpes","Ile-de-France","Occitanie","Hauts-de-France","Nouvelle-Aquitaine","Grand Est"],N)
anc=rng.integers(1,180,N)
nb_cred=rng.choice([1,2,3,4],N,p=[.55,.28,.12,.05])
montant_credit=np.round(rng.uniform(1500,35000,N),-1)
risk=rng.normal(0,1,N)+ (contrat=="Interim")*0.8+(contrat=="CDD")*0.5+(age<26)*0.5+(nb_cred>=3)*0.6-(anc>60)*0.4
# transactions
start=pd.Timestamp("2025-11-01"); end=pd.Timestamp("2026-11-01")
rows=[]; tid=1
p_late=1/(1+np.exp(-(risk-1.6)))
for i,c in enumerate(ids):
    k=rng.integers(6,13)
    dates=sorted(start+pd.to_timedelta(rng.integers(0,365,k),unit="D"))
    mens=round(montant_credit[i]/rng.choice([24,36,48,60]),2)
    for d in dates:
        u=rng.random()
        if u<p_late[i]*0.35: st,j="IMPAYE",int(rng.integers(30,120))
        elif u<p_late[i]: st,j="RETARD",int(rng.integers(1,30))
        else: st,j="PAYE",0
        rows.append([f"T{str(tid).zfill(6)}",c,d,mens,st,j]); tid+=1
tx=pd.DataFrame(rows,columns=["transaction_id","client_id","transaction_date","montant_echeance","statut_paiement","jours_retard"])
imp=tx[tx.statut_paiement!="PAYE"].groupby("client_id").montant_echeance.sum()
montant_impaye=np.round(pd.Series(ids).map(imp).fillna(0).values*rng.uniform(.6,1.4,N),2)
recent=tx[(tx.transaction_date>=end-pd.Timedelta(days=90))&(tx.statut_paiement!="PAYE")].groupby("client_id").size()
nrec=pd.Series(ids).map(recent).fillna(0).values
logit=-3.2+0.9*risk+0.55*nrec+6*(montant_impaye/revenu)
y=(rng.random(N)<1/(1+np.exp(-logit))).astype(int)
cl=pd.DataFrame(dict(client_id=ids,age=age,revenu_annuel=revenu,type_contrat=contrat,region=region,anciennete_mois=anc,nb_credits_actifs=nb_cred,montant_credit=montant_credit,montant_impaye=montant_impaye,defaut_paiement=y.astype(float)))
cl.loc[rng.random(N)<.08,"age"]=np.nan
new=rng.random(N)<.15
cl.loc[new,"defaut_paiement"]=np.nan
cl["defaut_paiement"]=cl.defaut_paiement.astype("Int64")
cl.to_csv("data/clients_db.csv",index=False)
tx["transaction_date"]=tx.transaction_date.dt.strftime("%d/%m/%Y")
tx.to_csv("data/transactions_csv.csv",index=False,sep=";")
print(len(cl),len(tx),cl.defaut_paiement.value_counts(dropna=False).to_dict(),cl.age.isna().sum())
