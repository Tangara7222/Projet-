import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stat
from Menu_Principal import load_data

# Initialisation de la variable de données
data1 = None

# Chargement du dataset
uploaded_file = st.file_uploader("Veuillez télécharger le dataset", type=["csv", "txt", "data"])
if uploaded_file is not None:
    data1 = load_data(uploaded_file)
else:
    st.error("Aucun fichier chargé")

# Analyse sur le dataset
if data1 is not None:
    col_num = data1.select_dtypes(include=['number']).columns.tolist()
    col_object = data1.select_dtypes(include='object').columns.tolist()

    st.header("Analyse Bivariée")
    quanti = col_num
    quali = col_object

    # 1. Aperçu des variables quantitatives
    st.subheader("1- Aperçu des variables quantitatives")
    pp_vars = st.multiselect("Veuillez sélectionner les variables", quanti)

    if pp_vars:
        try:
            pair_plot = sns.pairplot(data1[pp_vars])
            st.pyplot(pair_plot.figure)
        except Exception as e:
            st.error(f"Erreur lors de la création du pairplot : {str(e)}")

    # 2. Test de corrélation entre deux variables
    st.subheader("a- 🔗 Analyse de corrélation des variables")
    test_var = ["wheel-base", "length", "width", "height", "bore", "horsepower", "price"]

    botton_var1 = st.selectbox("Veuillez choisir la première variable", options=test_var)
    botton_var2 = st.selectbox("Veuillez choisir la deuxième variable", options=test_var)

    if botton_var1 and botton_var2 and botton_var1 != botton_var2:
        if botton_var1 in data1.columns and botton_var2 in data1.columns:
            statistic, pvalue = stat.spearmanr(data1[botton_var1], data1[botton_var2])
            if pvalue < 0.05 and statistic > 0.5:
                texte = f"**{botton_var1} et {botton_var2} sont corrélées avec {round(statistic, 3)} et un p-value de {round(pvalue, 3)}**"
                st.success(texte)
            else:
                texte = f"**{botton_var1} et {botton_var2} ne sont pas corrélées avec {round(statistic, 3)} et un p-value de {round(pvalue, 3)}**"
                st.error(texte)
        else:
            st.error("Les variables sélectionnées ne sont pas présentes dans le dataset.")

    # ANALYSE DES VARIABLES QUALITATIVES
    st.subheader("2 - Aperçu des variables qualitatives")
    label1 = ['fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
              'engine-type', 'num-of-cylinders', 'fuel-system']
    botton_graph = st.selectbox("Veuillez sélectionner une variable", options=label1)

    if botton_graph in data1.columns:
        tab = pd.crosstab(data1[botton_graph], data1['num-of-doors'])
        try:
            if tab.shape == (2, 2):
                statistic1, pvalue1 = stat.fisher_exact(tab)
                fig, ax = plt.subplots(figsize=(10, 10))
                title = f"{botton_graph} VS num-of-doors, fisher: {round(statistic1, 3)}, pvalue: {round(pvalue1, 3)}"
                ax.set_title(title)
                sns.countplot(x=botton_graph, hue='num-of-doors', data=data1)
                st.pyplot(fig)
            else:
                statistic0, pvalue0, dll, freq = stat.chi2_contingency(tab)
                fig, ax = plt.subplots(figsize=(10, 10))
                title = f"Répartition de {botton_graph} selon num-of-doors, chi2: {round(statistic0, 3)}, pvalue: {round(pvalue0, 3)}"
                ax.set_title(title)
                sns.countplot(x=botton_graph, hue='num-of-doors', data=data1)
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Erreur lors de l'analyse des variables qualitatives : {str(e)}")

    # 3. Aperçu des variables quantitatives et qualitatives
    st.subheader("3 - Aperçu des variables quantitatives et qualitatives")
    if "price" in data1.columns and botton_graph in data1.columns:
        fig, ax = plt.subplots(figsize=(18, 10))
        sns.boxplot(x=botton_graph, y="price", data=data1, ax=ax)
        ax.set_title(f"Boxplot de la variable {botton_graph} par rapport à 'price'")
        st.pyplot(fig)
    else:
        st.error("La variable 'price' ou la variable qualitative sélectionnée est absente du dataset.")

else:
    st.error("Base de données introuvable ! Veuillez réessayer")
