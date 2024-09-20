from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 2: Veri On Isleme
train_data, test_data, train_labels, test_labels = train_test_split(
    df['text'], df['category'], test_size=0.2, random_state=42
)
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Step 3: Word2Vec ile Tokenizasyon ve Kelime Gomme
preprocessed_data = [text_to_word_sequence(text) for text in train_data]
word2vec_model = Word2Vec(sentences=preprocessed_data, vector_size=128, window=5, min_count=1, workers=4)
word_index = {word: index for index, word in enumerate(word2vec_model.wv.index_to_key)}


# Sozlukteki her kelime icin Word2Vec yerlestirmelerini yazdirma
max_words_to_display = 1  
for word, index in word_index.items():
    if index < max_words_to_display:
        try:
            embedding_vector = word2vec_model.wv[word]
            print(f"Word: {word}, Embedding: {embedding_vector}")
        except KeyError:
            print(f"Word: {word} not found in Word2Vec embeddings.")


train_sequences_word2vec = [[word_index[word] for word in text_to_word_sequence(text) if word in word_index] for text in train_data]
test_sequences_word2vec = [[word_index[word] for word in text_to_word_sequence(text) if word in word_index] for text in test_data]

max_length = 100
train_padded_word2vec = pad_sequences(train_sequences_word2vec, maxlen=max_length, truncating='post')
test_padded_word2vec = pad_sequences(test_sequences_word2vec, maxlen=max_length, truncating='post')