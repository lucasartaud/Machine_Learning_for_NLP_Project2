import pandas as pd
import numpy as np
import streamlit as st

from huggingface_hub import hf_hub_download
from transformers import pipeline
from gensim.models import KeyedVectors

st.markdown(
    """
    <style>
        .block-container {
            max-width: 80vw;
            margin-left: auto;
            margin-right: auto;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Projet 2 NLP par Lucas Artaud et Iswarya Sivasubramaniam')

st.sidebar.header('Pages')
selected_page = st.sidebar.selectbox('Sélection de page', ['Résumé du texte', 'Prédiction des étoiles', 'Réponse aux questions'])
if selected_page == 'Résumé du texte':

    st.header('Résumé du texte', divider='blue')

    user_input = st.text_area("Entrez votre texte ici :", height=200)

    if st.button("Résumer"):
        if user_input:
            summarizer_pipeline = pipeline("summarization", model="plguillou/t5-base-fr-sum-cnndm")
            summary = summarizer_pipeline(user_input)
            st.subheader(f"Résumé du texte :")
            st.write(summary[0]['summary_text'])
        else:
            st.warning("Veuillez saisir du texte.")

if selected_page == 'Prédiction des étoiles':

    st.header('Prédiction des étoiles', divider='blue')

    user_input = st.text_area("Entrez votre texte ici :", height=200)

    if st.button("Prédire"):
        if user_input:
            predict_stars_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
            result = predict_stars_pipeline(user_input[:512])
            st.subheader(f"Nombre d'étoiles prédit :")
            st.write(result[0]['label'][0])
        else:
            st.warning("Veuillez saisir du texte.")

if selected_page == 'Réponse aux questions':

    st.header('Réponse aux questions', divider='blue')

    df = pd.read_csv('avis.csv')
    df['avis_vectorisé'] = df['avis_vectorisé'].apply(lambda x: np.array([float(item) for item in x.split(', ')]))

    user_question = st.text_input('Entrez votre texte ici :', height=200)
    if st.button('Soumettre'):
        if user_question:

            word2vec_model = KeyedVectors.load_word2vec_format(hf_hub_download(repo_id="Word2vec/wikipedia2vec_frwiki_20180420_300d", filename="frwiki_20180420_300d.txt"))
            def vectorize_sentence(sentence):
                words = sentence.split()  # Supposons que vos avis soient tokenisés par des espaces
                vectors = [word2vec_model[word] for word in words if word in word2vec_model.key_to_index]
                if vectors:
                    return sum(vectors) / len(vectors)
                else:
                    return None

            def cosine_similarity(embedding1, embedding2):
                return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

            def find_answer(question):
                vectorized_question = vectorize_sentence(question)
                similarities = []
                for index, row in df.iterrows():
                    similarities.append(cosine_similarity(vectorized_question, row['avis_vectorisé']))
                max_similarity_index = similarities.index(max(similarities))
                return df.iloc[max_similarity_index]

            answer = find_answer(user_question)
            st.subheader('Réponse trouvée :')
            st.write('Note :', answer['note'])
            st.write('Avis :', answer['avis'])
            st.write('Assureur :', answer['assureur'])
            st.write('Produit :', answer['produit'])
            st.write('Date de publication :', answer['date_publication'])
        else:
            st.warning("Veuillez saisir du texte.")
