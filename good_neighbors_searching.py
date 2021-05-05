from config import *
from image_func import *
from text_func import *
from cv_score import *
# To find the finest neighbors


def threshold_searching(df, imgtxt, embeddings,
                        LB=4.0, UB=6.0,
                        PRINT_CHUNK=False, metric='minkowski'):
    df1 = pd.DataFrame(columns=['target', 'pred_matches'])
    df1.target = df.target

    if imgtxt == 'img':
        if metric == 'cosine':
            thresholds = list(np.arange(LB, UB, 0.02))
        else:
            thresholds = list(np.arange(LB, UB, 0.2))
        scores = []
        for threshold in thresholds:
            _, image_predictions = get_image_neighbors(df, embeddings, threshold=threshold, metric=metric)
            df1.pred_matches = image_predictions
            MyCVScore = df1.apply(getMetric('pred_matches'), axis=1)
            score = MyCVScore.mean()
            print(f'CV score for threshold {threshold} = {score}')
            scores.append(score)
        thresholds_scores = pd.DataFrame({'thresholds': thresholds, 'scores': scores})
        max_score = thresholds_scores[thresholds_scores['scores'] == thresholds_scores['scores'].max()]
        best_threshold = max_score['thresholds'].values[0]
        best_score = max_score['scores'].values[0]
        print(f'Our best score is {best_score} and has a threshold {best_threshold}')

    elif imgtxt == 'txt':
        thresholds = list(np.arange(LB, UB, 0.02))
        scores = []
        for threshold in thresholds:
            text_predictions = get_text_predictions(df, embeddings, threshold=threshold, PRINT_CHUNK=PRINT_CHUNK)
            df1.pred_matches = text_predictions
            MyCVScore = df1.apply(getMetric('pred_matches'), axis=1)
            score = MyCVScore.mean()
            print(f'CV score for threshold {threshold} = {score}')
            scores.append(score)
        thresholds_scores = pd.DataFrame({'thresholds': thresholds, 'scores': scores})
        max_score = thresholds_scores[thresholds_scores['scores'] == thresholds_scores['scores'].max()]
        best_threshold = max_score['thresholds'].values[0]
        best_score = max_score['scores'].values[0]
        print(f'Our best score is {best_score} and has a threshold {best_threshold}')

    return best_threshold