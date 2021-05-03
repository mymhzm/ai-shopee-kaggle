from utils import *
from image_func import get_image_embeddings, get_image_neighbors

if __name__ == "__main__":

    seed_torch(CFG.seed)

    # 载入数据
    df, df_cu, image_paths = read_dataset(COMPUTE_CV)
    df.head()

    # 这里可以选择：1.直接使用embedding的结果
    #            2.重新对shopee的图片做embedding
    if not COMPUTE_CV:
        image_embeddings = get_image_embeddings(image_paths.values)
        if SAVE_IMGEMBEDDING:
            np.savetxt('image_embeddings_tf_efficientnet_b4_myown.csv', image_embeddings, delimiter=',')
    else:
        # 下面的路径是直接用的作者的embedding结果了，下载方式去config看看
        image_embeddings = np.loadtxt(
            './input/shopee-price-match-guarantee-embeddings/image_embeddings_tf_efficientnet_b4.csv', delimiter=',')

    # text部分
    # text_embeddings = get_text_embeddings(df_cu)

    # text和image都先走一下baseline
    if BASELINE_CHECKING:
        # text_predictions = get_text_predictions(df, text_embeddings)
        df, image_predictions = get_image_neighbors(df, image_embeddings, KNN=100 if len(df) > 3 else 3)

    print(df.head())
    print(image_predictions)

    # 下面无非就是在embedding之后，调下match模型（余弦相似度及knn）的超参了，就是作者那个search函数
