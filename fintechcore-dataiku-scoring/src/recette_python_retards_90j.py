# -*- coding: utf-8 -*-
# =====================================================================
# Recette Python Dataiku : retards de paiement sur 90 jours + jeu ML
# Entrée  : clients_transactions_prepared  (1 ligne par transaction)
# Sorties : clients_ml_ready      (clients avec defaut_paiement connu -> Lab)
#           clients_a_scorer      (clients sans cible -> recette Score)
# =====================================================================
import dataiku
import pandas as pd
import numpy as np

# ---- 1. Lecture de l'entrée --------------------------------------------
df = dataiku.Dataset("clients_transactions_prepared").get_dataframe()
df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce", utc=True)


# On force les colonnes numériques en nombres (les cases vides deviennent NaN).
cols_num = ["montant_echeance", "jours_retard", "age", "revenu_annuel",
            "anciennete_mois", "nb_credits_actifs", "montant_credit",
            "montant_impaye", "dette_ratio", "defaut_paiement"]
for c in cols_num:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Un retard = toute échéance non payée à temps (RETARD ou IMPAYE)
df["is_retard"] = (df["statut_paiement"] != "PAYE").astype(int)

# ---- 2. Window function : retards glissants sur 90 jours ---------------
# Pour chaque transaction, on compte les retards du même client
# dans la fenêtre [date - 90 jours ; date].
df = df.sort_values(["client_id", "transaction_date"])
df["nb_retards_90j_glissant"] = (
    df.set_index("transaction_date")
      .groupby("client_id")["is_retard"]
      .rolling("90D")
      .sum()
      .values
)

# ---- 3. Agrégation 1 ligne par client ---------------------------------
# Référence = dernière date connue dans les données (= "aujourd'hui" du flux)
date_ref = df["transaction_date"].max()
df_90 = df[df["transaction_date"] > date_ref - pd.Timedelta(days=90)]

agg_txn = df.groupby("client_id").agg(
    nb_transactions=("transaction_id", "count"),
    nb_retards_total=("is_retard", "sum"),
    jours_retard_moyen=("jours_retard", "mean"),
    jours_retard_max=("jours_retard", "max"),
    montant_echeance_moyen=("montant_echeance", "mean"),
    pic_retards_90j=("nb_retards_90j_glissant", "max"),
)
agg_90 = df_90.groupby("client_id").agg(
    nb_retards_90j=("is_retard", "sum"),
    nb_transactions_90j=("transaction_id", "count"),
)

cols_client = ["client_id", "age", "revenu_annuel", "type_contrat", "region",
               "anciennete_mois", "nb_credits_actifs", "montant_credit",
               "montant_impaye", "dette_ratio", "defaut_paiement"]
clients = df[cols_client].drop_duplicates("client_id").set_index("client_id")

out = clients.join(agg_txn).join(agg_90)
out[["nb_retards_90j", "nb_transactions_90j"]] = out[["nb_retards_90j", "nb_transactions_90j"]].fillna(0)
out["taux_retard_90j"] = np.where(out["nb_transactions_90j"] > 0,
                                  out["nb_retards_90j"] / out["nb_transactions_90j"], 0)
out = out.reset_index()

# ---- 4. Sous-ensembles nettoyés ---------------------------------------
# Pas d'identifiant en feature : client_id reste mais sera à rejeter dans le Lab
ml_ready  = out[out["defaut_paiement"].notna()].copy()
ml_ready["defaut_paiement"] = ml_ready["defaut_paiement"].astype(int)
a_scorer  = out[out["defaut_paiement"].isna()].drop(columns=["defaut_paiement"])

# ---- 5. Écriture des sorties ------------------------------------------
dataiku.Dataset("clients_ml_ready").write_with_schema(ml_ready)
dataiku.Dataset("clients_a_scorer").write_with_schema(a_scorer)
