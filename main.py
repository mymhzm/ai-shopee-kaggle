from text_func import get_text_embeddings, get_text_predictions
from utils import *
from image_func import get_image_embeddings, get_image_neighbors
from good_neighbors_searching import *
from cv_score import *

if __name__ == "__main__":
    seed_torch(CFG.seed)

    print("----------------data loading_start--------------")
    # 载入数据
    df, df_cu, image_paths = read_dataset(COMPUTE_CV)
    print(df.head())
    print("----------------data loading_end--------------\n")

    # image部分
    # 这里可以选择：1.直接使用embedding的结果
    #            2.重新对shopee的图片做embedding
    print("----------------image_embeddings_start--------------")
    if not COMPUTE_CV:
        image_embeddings = get_image_embeddings(image_paths.values)
        if SAVE_IMGEMBEDDING:
            np.savetxt('image_embeddings_tf_efficientnet_b4_myown.csv', image_embeddings, delimiter=',')
    else:
        print("----------------COMPUTE_CV:load embeddings--------------")
        # 下面的路径是直接用的作者的embedding结果了，下载方式去config看看
        image_embeddings = np.loadtxt(
            './input/shopee-price-match-guarantee-embeddings/image_embeddings_tf_efficientnet_b4.csv', delimiter=',')
    print("----------------image_embeddings_end--------------\n")


    # text部分
    print("----------------text_embeddings_start--------------")
    text_embeddings = get_text_embeddings(df_cu)
    print("----------------text_embeddings_end--------------\n")


    # text和image都先走一下baseline
    if BASELINE_CHECKING:
        print("----------------image_predictions--------------\n")
        df, image_predictions = get_image_neighbors(df, image_embeddings, KNN=100 if len(df) > 3 else 3)
        print("----------------text_predictions--------------\n")
        text_predictions = get_text_predictions(df, text_embeddings)

    #print(df.head())
    #print(image_predictions)

    # 下面无非就是在embedding之后，调下match模型（余弦相似度及knn）的超参了，就是作者那个search函数
    # 待搬运
    # 最后就是将text和image_predictions的值做一个拼接就好


    # Preparing Submission 准备提交的数据
    print("----------------submission_prepare_start--------------")
    df['image_predictions'] = image_predictions
    df['text_predictions'] = text_predictions
    tmp = df.groupby('image_phash').posting_id.agg('unique').to_dict()
    df['phash_predictions'] = df.image_phash.map(tmp)
    df['matches'] = df.apply(combine_predictions, axis=1)
    df[['posting_id', 'matches']].to_csv('submission.csv', index=False)
    print("----------------submission_prepare_end,please find the submission.csv--------------\n")

    # CV Score (BASELINE_CHECKING)
    print("----------------CV Score_start--------------")
    if COMPUTE_CV and BASELINE_CHECKING:
        df['matches_CV'] = df.apply(combine_for_cv, axis=1)
        tmp = df.groupby('label_group').posting_id.agg('unique').to_dict()
        df['target'] = df.label_group.map(tmp)
        MyCVScore = df.apply(getMetric('matches_CV'), axis=1)
        print('CV score =', MyCVScore.mean())
    elif COMPUTE_CV:
        tmp = df.groupby('label_group').posting_id.agg('unique').to_dict()
        df['target'] = df.label_group.map(tmp)
    print("----------------CV Score_end--------------\n")



    print("----------------Good Neighbors Searching_start--------------")
    #Good Neighbors Searching
    print("------Good Neighbors Searching_img----------")
    if COMPUTE_CV and NEIGHBORS_SEARCHING:
        best_threshold_img = threshold_searching(df, 'img', image_embeddings, LB=4.5, UB=4.6)
    print("------Good Neighbors Searching_txt----------")
    if COMPUTE_CV and NEIGHBORS_SEARCHING:
        best_threshold_txt = threshold_searching(df, 'txt', text_embeddings, LB=0.75, UB=0.76)
    print("------Good Neighbors Searching_img_cosine----------")
    if COMPUTE_CV and NEIGHBORS_SEARCHING and IMG_COSINE:
        best_threshold_img_cosine = threshold_searching(df, 'img', image_embeddings, LB=0.18, UB=0.19, metric='cosine')
    print("----------------Good Neighbors Searching_end--------------\n")

    print(f'COMPUTE_CV = {COMPUTE_CV}')
    print(f'NEIGHBORS_SEARCHING = {NEIGHBORS_SEARCHING}')
    print(f'BASELINE_CHECKING = {BASELINE_CHECKING}')
    print(f'SAVE_IMGEMBEDDING = {SAVE_IMGEMBEDDING}')
    print(f'THRES_METH = {THRES_METH}')
    print(f'STOP_WORDS = {STOP_WORDS}')
    print(f'IMG_COSINE = {IMG_COSINE}')

