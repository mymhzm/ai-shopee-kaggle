from utils import *

# Perform text embedding and turn characters into vectors
def get_text_embeddings(df_cu, max_features=25_000):
    model = TfidfVectorizer(stop_words=STOP_WORDS,
                            binary=True,
                            max_features=max_features)
    text_embeddings = model.fit_transform(df_cu['title']).toarray()
    return text_embeddings

# Use ‘Cosine Similarity Method’ to do text match
def get_text_predictions(df, embeddings, max_features=25_000, threshold=0.75, PRINT_CHUNK=True):
    print('Finding similar titles...')
    CHUNK = 1024 * 4
    CTS = len(df) // CHUNK
    if (len(df) % CHUNK) != 0:
        CTS += 1

    preds = []
    for j in range(CTS):
        a = j * CHUNK
        b = (j + 1) * CHUNK
        b = min(b, len(df))
        if PRINT_CHUNK:
            print('chunk', a, 'to', b)

        # Cosine similarity distance
        cts = np.matmul(embeddings, embeddings[a:b].T).T
        for k in range(b - a):
            IDX = np.where(cts[k,] > threshold)[0]
            o = df.iloc[IDX].posting_id.values #原本是cupy.asnumpy->np.asnumpy->IDX
            preds.append(o)

    return preds