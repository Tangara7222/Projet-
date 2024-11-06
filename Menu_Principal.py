import streamlit as st


import pandas as pd
import numpy as np
import seaborn as sns
import joblib
from sklearn.neighbors import KNeighborsRegressor




# Ajouter une image d'arrière-plan 
page_bg_img = ''' <style> [data-testid="stAppViewContainer"] { background-image: url("https://mobimg.b-cdn.net/v3/fetch/a4/a44b6eb557c05da374b1bcb7ffad8958.jpeg?w=1000&r=0.5625"); background-size: cover; } </style> ''' 
st.markdown(page_bg_img, unsafe_allow_html=True)
data = None
def main():
    st.header("**BIENVENU SUR NOTRE PLATFORME !**")
    st.title("ANALYSE ET PREDICTION DE DONNEES")
    st.subheader("Par Tangara Abdoulaye")

    choix = st.sidebar.radio("SousMenu",["Description","Documentation", "Prediction du prix"])

    if choix == "Description":
            st.write("""
                ### Description

                Cette application interactive, développée par Tangara Abdoulaye, est conçue pour vous aider à explorer, analyser et prédire des données de manière intuitive et efficace.

                **Fonctionnalités principales :**
                - **Téléchargement de données** : Importez facilement vos fichiers de données au format CSV, TXT ou DATA.
                - **Analyse exploratoire des données (EDA)** : Obtenez un aperçu complet des variables quantitatives et qualitatives de votre dataset grâce à des résumés statistiques et des visualisations graphiques.
                - **Nettoyage et traitement des données** : Remplacez les valeurs manquantes, convertissez les types de colonnes et préparez vos données pour l'analyse.
                - **Analyse bivariée** : Explorez les relations entre deux variables quantitatives grâce à des graphiques de corrélation et des tests statistiques.
                - **Graphiques interactifs** : Visualisez vos données avec des boxplots, histogrammes, diagrammes en barre et pie charts pour une compréhension approfondie.
                - **Prediction du prix** : Une prediction du prix basée sur le modèle KNN est également possible.
                     
                ### Base de donnée d'entrainement-test
                     
                **Ce jeu de données se compose de trois types d'entités** : 
                (a) la spécification d'une voiture en termes de diverses caractéristiques 
                (b) son classement de risque d'assurance attribué
                (c) ses pertes normalisées en utilisation par rapport à d'autres voitures.
                **La deuxième** évaluation correspond au degré auquel la voiture est plus risquée que son prix ne l'indique.
                Les voitures se voient initialement attribuer un symbole de facteur de risque associé à leur prix.
                Ensuite, si elles sont plus risquées (ou moins), ce symbole est ajusté en le déplaçant vers le haut (ou vers le bas) de l'échelle.
                Les actuaires appellent ce processus *"symboling"*. Une valeur de +3 indique que la voiture est risquée, -3 qu'elle est probablement assez sûre.

                **Le troisième facteur** est le paiement moyen relatif des pertes par année de véhicule assuré. Cette valeur est normalisée pour toutes les voitures dans une classification de taille particulière (petite deux portes, breaks, sport/spécialité, etc.), et représente la perte moyenne par voiture par an.

                -- Note : Plusieurs des attributs de la base de données pourraient être utilisés comme attributs de "classe".
                
            """)
            
    elif choix == "Documentation":
            st.write("""
            ### Documentation

            **Importation des bibliothèques :**
            ```python
            import streamlit as st
            import pandas as pd
            import numpy as np
            import seaborn as sns
            ```

            **Variable globale :**
            ```python
            data = None
            ```

            **Fonction principale :**
            ```python
            def main():
                st.header("**BIENVENUE SUR NOTRE PLATFORME !**")
                st.title("ANALYSE ET PREDICTION DE DONNEES")
                st.subheader("Par Tangara Abdoulaye")

                choix = st.sidebar.radio("SousMenu", ["Description", "Documentation"])
                st.sidebar.text("Presentation de l'auteur")
            ```

            Cette fonction configure l'interface utilisateur principale de l'application Streamlit.

            **Exécution de la fonction principale :**
            ```python
            if __name__ == '__main__' : 
                main()
            ```

            **Fonction de charge et de nettoyage des données :**
            ```python
            def load_data(uploaded_file):
                columns = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
                        'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
                        'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower',
                        'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
                data = pd.read_csv(uploaded_file, delimiter=",", names=columns)
                if not data.empty:
                    data = data.replace("?", np.nan)
                    columns_to_convert = [col for col in data.columns if col != 'num-of-doors']
                    for col in columns_to_convert:
                        data[col] = pd.to_numeric(data[col], errors='ignore')
                    data['num-of-doors'] = data['num-of-doors'].astype(object)
                    data['normalized-losses'] = data['normalized-losses'].fillna(data['normalized-losses'].mean())
                    data = data.dropna(axis='index')
                    return data
                else:
                    return None
            ```

            Cette fonction charge les données à partir d'un fichier téléchargé par l'utilisateur, remplace les valeurs manquantes, convertit les types de colonnes appropriées et remplit les valeurs manquantes avec la moyenne pour certaines colonnes. Enfin, elle retourne le jeu de données nettoyé.
        """)
    elif choix == "Prediction du prix":
      # CHARGEMENT DES MODULES
        st.title("LA PREDICTION DU PRIX DUNE VOITURE")
        st.subheader("Par : Abdoulaye Tangara")
        st.write("")
        # CHARGEMENT DU MODELE
        model = joblib.load("model_prediction")

        # FOCNTION INFERENECE
        def inference(normalized_losses, wheel_base, length, width, height, bore,
            stroke, compression_ratio, horsepower, peak_rpm):
            features = np.array([
                normalized_losses, wheel_base, length, width, height, bore,
            stroke, compression_ratio, horsepower, peak_rpm
                ])
            prediction = model.predict(features.reshape(1,-1))
            return prediction

        # Predictions
        normalized_losses = st.number_input("normalized-losses", value = 0)
        wheel_base = st.number_input("wheel-base", value = 0)
        length = st.number_input("length", value = 0,min_value=0)
        width = st.number_input("width", value = 0,min_value=0)
        height = st.number_input("height", value = 0,min_value=0)
        bore = st.number_input("bore", value = 0)
        stroke = st.number_input("stroke", value = 0)
        compression_ratio = st.number_input("compression-ratio", value = 0)
        horsepower = st.number_input("horsepower", value = 0)
        peak_rpm = st.number_input("peak-rpm", value = 0)

        nom = st.text_input(label = "VEUILLEZ ENTRER VOTRE NOM SVP ! ")

        # LE BOUTTON DE PREDICTION

        if st.button("Predire"):
            prediction = inference(
                normalized_losses,wheel_base,length,width,height,
                bore,stroke,compression_ratio,horsepower,peak_rpm
                )
            resultat = f"Merci M./Mme : {nom}, vous pourrez donc avoir une voiture de : {prediction[0]} $"
            st.success(resultat)
    else:
        st.error("Erreur survenue lors de l'opération, veuillez reprendre s'il vous plaît.")
            


if __name__ == '__main__' : 
    main()


# Fonction de charge et de nettoyage des données
def load_data(uploaded_file):
    columns = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
               'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
               'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower',
               'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
    data = pd.read_csv(uploaded_file, delimiter=",", names=columns)
    if not data.empty:
        data = data.replace("?", np.nan)
        columns_to_convert = [col for col in data.columns if col != 'num-of-doors']
        for col in columns_to_convert:
            data[col] = pd.to_numeric(data[col], errors='ignore')
        data['num-of-doors'] = data['num-of-doors'].astype(object)
        data['normalized-losses'] = data['normalized-losses'].fillna(data['normalized-losses'].mean())
        data = data.dropna(axis='index')
        return data
    else:
        return None
    


   
