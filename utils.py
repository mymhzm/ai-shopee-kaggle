from config import *
import os
import cv2
import math
import random
from tqdm import tqdm
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import torch
import timm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
import gc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def read_dataset(COMPUTE_CV):
    if COMPUTE_CV:
        df = pd.read_csv('./input/shopee-product-matching/train.csv')
        df_cu = pd.DataFrame(df)
        image_paths = './input/shopee-product-matching/train_images/' + df['image']

    else:
        df = pd.read_csv('./input/shopee-product-matching/test.csv')
        df_cu = pd.DataFrame(df)
        image_paths = './input/shopee-product-matching/test_images/' + df['image']

    return df, df_cu, image_paths

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def combine_predictions(row):
    x = np.concatenate([row['image_predictions'], row['text_predictions'], row['phash_predictions']])
    return ' '.join( np.unique(x) )

def combine_predictions_cosine(row):
    x = np.concatenate([row['image_predictions'], row['text_predictions'], row['phash_predictions'], row['image_predictions_cosine']])
    return ' '.join( np.unique(x) )

def combine_for_cv(row):
    x = np.concatenate([row['image_predictions'], row['text_predictions'], row['phash_predictions']])
    return np.unique(x)

def combine_for_cv_cosine(row):
    x = np.concatenate([row['image_predictions'], row['text_predictions'], row['phash_predictions'], row['image_predictions_cosine']])
    return np.unique(x)