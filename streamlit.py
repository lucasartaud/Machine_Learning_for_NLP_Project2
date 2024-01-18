import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import language_tool_python
import tensorflow_hub as hub
import streamlit as st

from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense

from nltk import word_tokenize
from nltk.util import ngrams

from collections import Counter
from huggingface_hub import hf_hub_download
from transformers import pipeline
from gensim.models import KeyedVectors

pd.set_option('display.max_colwidth', 300)

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

st.title('NLP Project 2 by Lucas Artaud and Iswarya Sivasubramaniam')

folder = 'Traduction avis clients'
df = pd.DataFrame()
for i in range(1, 36):
    filename = f'avis_{i}_traduit.xlsx'
    path = os.path.join(folder, filename)
    df = pd.concat([df, pd.read_excel(path)], ignore_index=True)

df.drop(columns=['type', 'avis_en', 'avis_cor', 'avis_cor_en'], inplace=True)
df['avis'] = df['avis'].replace('\n', ' ', regex=True)
df.dropna(inplace=True)
df['note'] = df['note'].astype(int)

st.sidebar.header('Pages')
selected_page = st.sidebar.selectbox('Select a page', ['Welcome', 'Data cleaning, summary, translation and sentiment analysis', 'Word2Vec', 'Supervised Learning'])
if selected_page == 'Welcome':

    st.write('Welcome to this page. Please select a page.')

if selected_page == 'Data cleaning, summary, translation and sentiment analysis':

    st.header('Data cleaning', divider='blue')

    st.subheader('Highlighting frequent words (and n-grams)', divider='violet')
    text = ' '.join(df['avis'])
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha()]
    top_words = Counter(tokens).most_common(10)
    st.write('10 most frequent words:')
    st.dataframe(top_words)

    bigrams = list(ngrams(tokens, 2))
    top_bigrams = Counter(bigrams).most_common(10)
    st.write('10 most frequent bigrams:')
    st.dataframe(top_bigrams)

    st.subheader('Spelling correction', divider='violet')
    df_subset = df.head(10)
    tool = language_tool_python.LanguageTool('fr')
    def spelling_correction_2(text):
        return tool.correct(text)
    df_subset['avis_corriges'] = df_subset['avis'].apply(spelling_correction_2)
    st.dataframe(df_subset)

    st.header('Summary and translation', divider='blue')

    st.subheader('Summary', divider='violet')
    summarizer_pipeline = pipeline("summarization", model="plguillou/t5-base-fr-sum-cnndm", max_length=80)
    def summarizer(text):
        result = summarizer_pipeline(text)
        return result[0]['summary_text']
    df_subset['résumé'] = df_subset['avis_corriges'].apply(summarizer)
    st.dataframe(df_subset)
    st.write('Some reviews are too short to summarize')

    st.subheader('Translation', divider='violet')
    fr_en_translator_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
    def fr_en_translator(text):
        result = fr_en_translator_pipeline(text)
        return result[0]['translation_text']
    df_subset['traduction'] = df_subset['avis_corriges'].apply(fr_en_translator)
    st.dataframe(df_subset)

    st.header('Sentiment analysis', divider='blue')

    st.subheader('Predict positive or negative', divider='violet')
    predict_pos_neg_pipeline = pipeline("sentiment-analysis")
    def predict_pos_neg(text):
        result = predict_pos_neg_pipeline(text)
        return result[0]['label']
    df_subset['pos_neg_pred'] = df_subset['avis'].apply(predict_pos_neg)
    st.dataframe(df_subset)

    st.subheader('Predict star', divider='violet')
    predict_stars_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    def predict_stars(text):
        result = predict_stars_pipeline(text[:512])
        return result[0]['label'][0]
    df_subset['stars_pred'] = df_subset['avis'].apply(predict_stars)
    st.dataframe(df_subset)

if selected_page == 'Word2Vec':

    st.header('Word2Vec', divider='blue')

    st.subheader('Vectorise reviews', divider='violet')
    word2vec_model = KeyedVectors.load_word2vec_format(hf_hub_download(repo_id="Word2vec/wikipedia2vec_frwiki_20180420_300d", filename="frwiki_20180420_300d.txt"))
    def vectorize_sentence(sentence):
        words = sentence.split()  # Supposons que vos avis soient tokenisés par des espaces
        vectors = [word2vec_model[word] for word in words if word in word2vec_model.key_to_index]
        if vectors:
            return sum(vectors) / len(vectors)
        else:
            return None
    df['avis_vectorisé'] = df['avis'].apply(vectorize_sentence)
    st.dataframe(df)

    st.subheader('Word2Vec Training', divider='violet')
    df.dropna(inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df['avis_vectorisé'], df['note'], test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train.tolist(), y_train)
    y_pred = model.predict(X_test.tolist())
    y_pred = np.round(y_pred).astype(int)
    st.write('RMSE =', mean_squared_error(y_test, y_pred, squared=False))
    st.write('R² =', r2_score(y_test, y_pred))

    st.subheader('Visualization of embeddings with Matplotlib', divider='violet')
    embeddings = np.array(df['avis_vectorisé'].tolist())
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    fig, ax = plt.subplots()
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=df['note'])
    ax.set_title("Visualization of Embeddings")
    st.pyplot(fig)

    st.subheader('Implementation of Euclidean distance and cosine similarity', divider='violet')
    def euclidean_distance(embedding1, embedding2):
        return np.linalg.norm(embedding1 - embedding2)
    def cosine_similarity(embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    euclidean_distances = []
    cosine_similarities = []
    for i in range(1000):
        for j in range(i + 1, 1000):
            euclidean_distances.append(euclidean_distance(embeddings_2d[i], embeddings_2d[j]))
            cosine_similarities.append(cosine_similarity(embeddings_2d[i], embeddings_2d[j]))
    st.write('Average euclidean distance for 1000 first embeddings =', np.mean(euclidean_distances))
    st.write('Average cosine similarity for 1000 first embeddings =', np.mean(cosine_similarities))

    st.subheader('Question answering with semantic search', divider='violet')
    def find_answer(question):
        vectorized_question = vectorize_sentence(question)
        similarities = []
        for index, row in df.iterrows():
            similarities.append(cosine_similarity(vectorized_question, row['avis_vectorisé']))
        max_similarity_index = similarities.index(max(similarities))
        return df.iloc[max_similarity_index]
    st.write(find_answer("Je veux une assurance auto qui ne coûte pas trop cher"))

    user_question = st.text_input('Ask your question (in French). Example "Je veux une assurance auto qui ne coûte pas trop cher":')
    if st.button('Submit'):
        if user_question:
            answer = find_answer(user_question)
            st.subheader('Answer found:')
            st.write(answer)
        else:
            st.warning("Please enter a question before submitting.")

if selected_page == 'Supervised Learning':

    st.header('Supervised Learning', divider='blue')

    st.subheader('TF-IDF', divider='violet')
    tfidf_vectorizer = TfidfVectorizer(min_df=0.1, max_df=0.3)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['avis'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    tfidf_df = pd.concat([df.reset_index(drop=True)['avis'], tfidf_df], axis=1)
    st.dataframe(tfidf_df)
    st.write('We set a max_df to ignore words that are too frequent in the French language and of no interest, and we set a min_df to ignore words that are not frequent enough in our opinions.')

    st.subheader('Basic model with an embedding layer', divider='violet')
    X = tfidf_df.drop('avis', axis=1)
    y = df.reset_index(drop=True)['note']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Sequential()
    model.add(Embedding(input_dim=len(X.columns), output_dim=X.shape[1], input_length=len(X.columns)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    loss, mae = model.evaluate(X_test, y_test)
    st.write('Loss =', loss)
    st.write('Mean Absolute Error =', mae)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].plot(history.history['mae'], label='Training MAE')
    axes[0].plot(history.history['val_mae'], label='Validation MAE')
    axes[0].set_title('Mean Absolute Error (MAE)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MAE')
    axes[0].legend()
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.subheader('Universal Sentence Embedding', divider='violet')
    embed = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/universal-sentence-encoder/versions/2")
    X = embed(df['avis']).numpy()
    y = df.reset_index(drop=True)['note'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(512,)))  # 512 is the dimensionality of Universal Sentence Embeddings
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    loss, mae = model.evaluate(X_test, y_test)
    st.write('Loss =', loss)
    st.write('Mean Absolute Error =', mae)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].plot(history.history['mae'], label='Training MAE')
    axes[0].plot(history.history['val_mae'], label='Validation MAE')
    axes[0].set_title('Mean Absolute Error (MAE)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MAE')
    axes[0].legend()
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
