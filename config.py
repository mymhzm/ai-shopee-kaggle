# 运行参数控制脚本
import pandas as pd
import torch

pd.set_option('display.max_columns', 1000000)   # 可以在大数据量下，没有省略号
pd.set_option('display.max_rows', 1000000)
pd.set_option('display.max_colwidth', 1000000)
pd.set_option('display.width', 1000000)


COMPUTE_CV = True # 是否只是进行cv计算（进行cv计算需要已有embedding）
NEIGHBORS_SEARCHING = True
SAVE_IMGEMBEDDING = True # 是否保存embedding结果
BASELINE_CHECKING = True # 是否走一下baseline看看效果
THRES_METH = 'BOOM' # 'BOOM', 'BOOM_OPTIMIZED', 'THRES', 'THRES_OPTIMIZED'
STOP_WORDS = None # 'english', None
IMG_COSINE = True

df = pd.read_csv('./input/shopee-product-matching/test.csv')
if len(df)>3:
    COMPUTE_CV = False
if COMPUTE_CV:
    print('this submission notebook will compute CV score but commit notebook will not')
else:
    print('this submission notebook will only be used to submit result')

class CFG:
    seed = 54
    classes = 11014
    scale = 30
    margin = 0.5
    model_name = 'tf_efficientnet_b4'
    fc_dim = 512
    img_size = 512
    batch_size = 20
    num_workers = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 作者提供的预训练模型
    # https://www.kaggle.com/chienhsianghung/eff-b4-tfidf-w-cv-for-threshold-searching
    # 拉到最下，找到这个finetune好的下载
    model_path = './model_saved/arcface_512x512_tf_efficientnet_b4_LR.pt'

    # 网上download的pretrained模型.pt .pth等
    #model_path = '../model_saved/xxxxxxx.pt'