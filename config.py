# Run parameter control
import pandas as pd
import torch


pd.set_option('display.max_columns', 1000000)
pd.set_option('display.max_rows', 1000000)
pd.set_option('display.max_colwidth', 1000000)
pd.set_option('display.width', 1000000)


COMPUTE_CV = True # Whether it is only for cv calculation (cv calculation requires embedding that has been trained)
NEIGHBORS_SEARCHING = True
SAVE_IMGEMBEDDING = True # Whether to save the embedding result
BASELINE_CHECKING = True # Whether to take a look at the baseline to see the effect
THRES_METH = 'BOOM' # 'BOOM', 'BOOM_OPTIMIZED', 'THRES', 'THRES_OPTIMIZED'
STOP_WORDS = None # 'english', None
IMG_COSINE = True

class CFG:
    seed = 54
    classes = 11014
    scale = 30
    margin = 0.5

    model_name = 'tf_efficientnet_b4' #Pre-trained model name
    fc_dim = 512
    img_size = 512
    batch_size = 20
    num_workers = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = './model_saved/arcface_512x512_tf_efficientnet_b4_LR.pt' #Pre-training modelpath