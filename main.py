from utils import *
from image_func import get_image_embeddings, get_image_neighbors

if __name__ == "__main__":
    # 载入数据
    df, df_cu, image_paths = read_dataset(COMPUTE_CV)
    df.head()

    # 这里可以直接使用embedding的结果
    if not COMPUTE_CV:
        image_embeddings = get_image_embeddings(image_paths.values)
        if SAVE_IMGEMBEDDING: np.savetxt('image_embeddings_tf_efficientnet_b4_myown.csv', image_embeddings,
                                         delimiter=',')
    else:
        image_embeddings = np.loadtxt(
            './input/shopee-price-match-guarantee-embeddings/image_embeddings_tf_efficientnet_b4.csv', delimiter=',')

    # text部分
    # text_embeddings = get_text_embeddings(df_cu)

    # text和image都先走一下baseline
    # if BASELINE_CHECKING:
    #     # text_predictions = get_text_predictions(df, text_embeddings)
    #     df, image_predictions = get_image_neighbors(df, image_embeddings, KNN=100 if len(df) > 3 else 3)
    #     df.head()

