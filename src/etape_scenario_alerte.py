# -*- coding: utf-8 -*-
# =====================================================================
# Étape "Execute Python code" du Scénario Dataiku
# Compte les clients à haut risque et le met dans une variable
# réutilisable dans le Reporter : ${nb_haut_risque}
# =====================================================================
import dataiku
from dataiku.scenario import Scenario

SEUIL = 0.5   # probabilité de défaut au-delà de laquelle un client est "haut risque"

df = dataiku.Dataset("clients_a_scorer_scored").get_dataframe()

nb_haut_risque = int((df["proba_1"] >= SEUIL).sum())
nb_total = len(df)

Scenario().set_scenario_variables(
    nb_haut_risque=nb_haut_risque,
    nb_clients_scores=nb_total,
    seuil_risque=SEUIL,
)
print(f"{nb_haut_risque} clients à haut risque sur {nb_total}")
