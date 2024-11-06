import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from PyPDF2 import PdfMerger
st.set_page_config(page_title='Modilo', page_icon='🚀') 

# from Home import data
col_object = None
col_num = None
data1 = None
data = None

# Titre de l'application
st.title("Bienvenue sur la page analyse de donnée !")
st.header("📊 Analyse univariée des données")

# Widget de téléchargement de fichier
# Chargement de la base
# Télécharger le dataset
uploaded_file = st.file_uploader(" 📁 Veuillez télécharger le dataset", type=["csv","txt","data"])

# Vérification si un fichier a été téléchargé
if uploaded_file is not None:
    # Lecture du fichier en tant que DataFrame Pandas
    ## La liste des colonnes
    columns = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
        'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower',
        'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
    data = pd.read_csv(uploaded_file, delimiter= ",", names = columns)
    if not data.empty:
        st.markdown("📋 Tableau de données")

        st.write(data)
else:
    st.info("ℹ️ Information importante") 
    st.error("❌ Erreur : Le dataset n'a pas été chargé.")

# Afficher la dimensionnalité du dataset
if data is not None:
    st.markdown(f"**Votre dataset contient exactement {data.shape[0]} lignes et {data.shape[1]} colonnes**") 
# ANALYSE EXPLORATOIRE
st.subheader("📈 Résumé statistique")

## Analyse des variables quantitatives
st.subheader("1 - Analyse global des variables quantitatives")
if data is not None:
    st.write(data.describe())

## Analyse des variables qualitatives
st.subheader("2 - Analyse global des variables qualitatives")
if data is not None:
    st.write(data.describe(include='object'))

#######################################################################################################


# Vérifiez si `data` est chargé correctement avant d'exécuter des opérations
if data is not None:
    # Remplacer toutes les occurrences de "?" par np.nan dans l'ensemble des données
    data = data.replace("?", np.nan)

    # Conversion des colonnes pertinentes en types numériques, sauf 'num-of-doors'
    columns_to_convert = [col for col in data.columns if col != 'num-of-doors']
    for col in columns_to_convert:
        # Convertir la colonne en float si possible
        data[col] = pd.to_numeric(data[col], errors='ignore')

    # Traiter 'num-of-doors' séparément comme une variable catégorielle
    data['num-of-doors'] = data['num-of-doors'].astype(object)

    # Remplir les valeurs manquantes avec la moyenne pour "normalized-losses"
    data['normalized-losses'] = data['normalized-losses'].fillna(data['normalized-losses'].mean())

    # Supprimer les lignes restantes qui contiennent des valeurs manquantes
    if data is not None:
        data1 = data.dropna(axis='index')

    # Afficher un message de succès
    st.success(" 🛠️ Les données ont été traitées avec succès. Vous disposez desormais des données prêtes à l'AED et à la modélisation")
    st.write(data1.head())
else:
    st.error(" ❌ Le dataset n'a pas été chargé. Veuillez vérifier que le fichier est bien importé.")


#################################################################################################

st.subheader(" Observation des outliers")

# Les boxplots des variables quantitatives
if data1 is not None:
    col_num = data1.select_dtypes(include='float').columns.tolist()
    option0 = st.selectbox("Veuillez selectionner une variable", col_num)

    fig, ax = plt.subplots(figsize=(10, 10))
    title = f"Le boxplot de {option0}"
    ax.set_title(title)
    sns.boxplot(data1[option0], ax=ax)  # Ajout de `x=`
    st.pyplot(fig)
    plt.savefig(title + ".pdf")

    st.subheader("📊 Analyse des variables quantitatives")
    # Analyse de normalité des variables quantitatives

    fig, ax = plt.subplots(figsize=(10, 10))
    statistic, pvalue = stats.shapiro(data1[option0])
    if pvalue > 0.05:
        label = f"L'histogram de {option0}, avec W : {round(statistic, 3)} et p-value : {round(pvalue, 3)}. Donc suit une Dist. Normale"
    else:
        label = f"L'histogram de {option0}, avec W : {round(statistic, 3)} et p-value : {round(pvalue, 3)} Donc ne suit pas une Dist. Normale"
    ax.set_title(label)
    sns.histplot(x=data1[option0], kde=True, ax=ax)  # Ajout de `x=`
    st.pyplot(fig)
    plt.savefig(f"hist-{option0}.pdf")

    st.subheader("🔠 Analyse des variables qualitatives")

    # Le bar chart pour les variables qualitatives
    col_object1 = data1.select_dtypes(include='object').columns.tolist()
    option1 = st.selectbox("Veuillez selectionner une variable", col_object1)

    st.markdown("**Diagramme en Barre**")
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.countplot(x=option1, data=data1, ax=ax)
    
    label = f"Diagramme en barre de {option1}"
    ax.set_title(label)  # Correction de `sns.countplot()`
    st.pyplot(fig)
    plt.savefig(f"Count-{option1}.pdf")

    # Le pie chart pour les variables qualitatives
    option2 = option1
    if option2:
        value_count = data1[option2].value_counts()
        labels = value_count.index
        st.markdown("**Diagramme circulaire**")
        fig, ax = plt.subplots(figsize=(10, 10))
        title = f"Le Pie Chart de {option2}"
        ax.set_title(title)
        ax.pie(value_count, labels=labels, autopct='%1.1f%%')
        st.pyplot(fig)
        plt.savefig(f"piechart-{option2}.pdf")
        if "graph" not in st.session_state:
            st.session_state.graph = list()
        files_name= [title + ".pdf","hist-{option0}.pdf","Count-{option1}.pdf","piechart-{option2}.pdf"]
        file_merged = PdfMerger()
        for file in files_name:
            file_merged.append(file)
            output_filename = "merged_output.pdf"
            file_merged.write(output_filename)

    # Fermeture de l'objet PdfMerger pour libérer les ressources
        file_merged.close()

    # Fermeture des figures matplotlib
        plt.close()
else:
    st.error("Le dataset n'a pas été chargé. Veuillez vérifier que le fichier est bien importé.")


