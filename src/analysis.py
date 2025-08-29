from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def analyze_topics(docs, n_topics=5):
    texts = [doc.text for doc in docs]

    # German stop words
    german_stop_words = [
        'der', 'die', 'das', 'ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr', 'sie',
        'und', 'oder', 'aber', 'in', 'auf', 'an', 'zu', 'von', 'mit', 'bei', 'für',
        'war', 'ist', 'bin', 'sind', 'haben', 'hat', 'hatte', 'wird', 'kann', 'soll',
        'mich', 'mir', 'mein', 'meine', 'sich', 'sein', 'seine', 'nicht', 'auch',
        'wenn', 'dann', 'wie', 'was', 'wo', 'wann', 'warum', 'heute', 'gestern',
        'am', 'im', 'zum', 'zur', 'ein', 'eine', 'einen', 'einer', 'eines', 'dem',
        'den', 'des', 'sehr', 'noch', 'nur', 'schon', 'mehr', 'etwas', 'nichts',
        'alle', 'alles', 'viel', 'wenig', 'gut', 'schlecht', 'gross', 'klein', 'habe', 'uns',
        'da', 'somit', 'wieder', 'weiter', 'dass', 'jetzt', 'so', 'als',

                # Additional filler words from your results
        'wirklich', 'nach', 'zeit', 'über', 'doch', 'waren', 'um', 'dabei',
        'eigentlich', 'werden', 'man', 'diese', 'durch', 'immer', 'nun', 'einem',
        'konnte', 'jedoch', 'werde', 'morgen', 'doch', 'leben', 'tägliches',

         'dieses', 'letzten', 'sollte', 'derzeit', 'kein', 'darauf', 'meinem',
        'gerade', 'wurde', 'beim', 'will', 'gemacht', 'jahr', 'tage', 'dieser',
        'keine', 'meinen', 'hier', 'macht', 'meiner', 'ihm', 'musste', 'mal',
        'ziemlich', 'leider', 'später', 'dort', 'machte', 'erste', 'redete',
        'tag', 'fuhr', 'abend', 'vor', 'erst', 'aus', 'gar', 'machten', 'traf',
        'echt', 'gehen', 'wollte', 'weg', 'gegen', 'voll', 'redeten', 'einmal',
        'einfach', 'hatten', 'fühlte', 'woche', 'ging', 'kam', 'schön', 'muss',
        'machen', 'anderen', 'wäre', 'würde', 'paar', 'denke', 'ob', 'vielleicht',
        'stunden', 'sondern', 'damit', 'mal', 'beim', 'ziemlich',

        # English words that slipped in
        'the', 'to', 'and', 'of', 'a', 'is', 'it', 'in', 'for', 'on', 'with',
        'as', 'at', 'by', 'from', 'this', 'that', 'was', 'are', 'been', 'have',
        'had', 'has', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'learning', 'review',

        # HTML/Markdown artifacts
        'figure', 'img', 'src', 'width', 'res', 'height', 'alt', 'png', 'jpg',
        'jpeg', 'gif', 'class', 'style', 'div', 'span', 'href', 'link', 'url',

        # Numbers and common measurements
        '2024', '2025', '2023', '2022', '300', '400', '500', '100', '200',
        '10', '20', '30', '40', '50', '60', '70', '80', '90',

        # Common time/frequency words
        'morgens', 'mittags', 'abends', 'nachts', 'täglich', 'wöchentlich',
        'montag', 'dienstag', 'mittwoch', 'donnerstag', 'freitag', 'samstag', 'sonntag'

    ]

    vectorizer = TfidfVectorizer(
        max_features=200,
        stop_words=german_stop_words,
        min_df=2,
        max_df=0.7,
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(texts)



    kmeans = KMeans(n_clusters=n_topics, random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)
    feature_names = vectorizer.get_feature_names_out()
    cluster_info = {}

    for i in range(n_topics):

        center = kmeans.cluster_centers_[i]
        top_indices = center.argsort()[-15:][::-1]  # Top 15, reversed
        top_terms = [feature_names[idx] for idx in top_indices]

        # Count documents in this cluster
        doc_count = np.sum(clusters == i)

        cluster_info[i] = {
            'terms': top_terms,
            'doc_count': doc_count,
            'documents': [j for j, c in enumerate(clusters) if c == i]
        }

        print(f"Topic {i} ({doc_count} documents): {', '.join(top_terms[:10])}")

    return clusters, cluster_info

