from text_func import get_text_embeddings, get_text_predictions
from utils import *
from image_func import get_image_embeddings, get_image_neighbors
from good_neighbors_searching import *
from cv_score import *

if __name__ == "__main__":
    seed_torch(CFG.seed)

    print("----------------data loading_start--------------")
    df, df_cu, image_paths = read_dataset(COMPUTE_CV)
    print(df.head())
    print("----------------data loading_end--------------\n")

    # image embedding
    # Here you can choose: 1. Use the result of embedding directly
    #                      2. Re-embedding the pictures in the shopee dataset
    print("----------------image_embeddings_start--------------")
    if not COMPUTE_CV:
        image_embeddings = get_image_embeddings(image_paths.values)
        if SAVE_IMGEMBEDDING:
            np.savetxt('./input/shopee-price-match-guarantee-embeddings/image_embeddings_tf_efficientnet_b4.csv', image_embeddings, delimiter=',')
    else:
        print("----------------COMPUTE_CV:load embeddings--------------")
        # Use trained embedding for image classification
        image_embeddings = np.loadtxt(
            './input/shopee-price-match-guarantee-embeddings/image_embeddings_tf_efficientnet_b4.csv', delimiter=',')
    print("----------------image_embeddings_end--------------\n")

    # text embedding
    print("----------------text_embeddings_start--------------")
    text_embeddings = get_text_embeddings(df_cu)
    print("----------------text_embeddings_end--------------\n")

    # Model prediction
    print("----------------image_predictions--------------\n")
    df, image_predictions = get_image_neighbors(df, image_embeddings, KNN=100 if len(df) > 3 else 3)
    print("----------------text_predictions--------------\n")
    text_predictions = get_text_predictions(df, text_embeddings)

    #print(df.head())
    #print(image_predictions)

    #Output result: According to the prediction results of image, text and phash,
    #perform image match to obtain the closest image set of each image
    print("----------------submission_prepare_start--------------")
    df['image_predictions'] = image_predictions
    df['text_predictions'] = text_predictions
    tmp = df.groupby('image_phash').posting_id.agg('unique').to_dict()
    #Perceptual hash
    df['phash_predictions'] = df.image_phash.map(tmp)
    df['matches'] = df.apply(combine_predictions, axis=1)
    df[['posting_id', 'matches']].to_csv('submission.csv', index=False)
    print("----------------submission_prepare_end,please find the submission.csv--------------\n")

    # CV Score
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

    #Optimization: find the best threshold
    print("----------------Good Neighbors Searching_start--------------")
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

    #Output the result of the above execution
    print(f'COMPUTE_CV = {COMPUTE_CV}')
    print(f'NEIGHBORS_SEARCHING = {NEIGHBORS_SEARCHING}')
    print(f'BASELINE_CHECKING = {BASELINE_CHECKING}')
    print(f'SAVE_IMGEMBEDDING = {SAVE_IMGEMBEDDING}')
    print(f'THRES_METH = {THRES_METH}')
    print(f'STOP_WORDS = {STOP_WORDS}')
    print(f'IMG_COSINE = {IMG_COSINE}')