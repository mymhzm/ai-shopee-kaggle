from utils import *
from data import ShopeeDataset

# 模型结构
# embedding方法
# ArcMarginProduct分类方法

# 构建上层分类NN，只会在pretrainned或者finetuning时生效,
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.ls_eps = ls_eps
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output, nn.CrossEntropyLoss()(output, label)


# 构建EFF embedding结构
class ShopeeModel(nn.Module):

    def __init__(
            self,
            n_classes=CFG.classes,
            model_name=CFG.model_name,
            fc_dim=CFG.fc_dim,
            margin=CFG.margin,
            scale=CFG.scale,
            use_fc=True,
            pretrained=True):

        super(ShopeeModel, self).__init__()
        print('Building Model Backbone for {} model'.format(model_name))

        # 通过timm lib载入指定模型结构
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.use_fc = use_fc

        if use_fc:
            self.dropout = nn.Dropout(p=0.1)
            self.classifier = nn.Linear(in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            in_features = fc_dim

        # 做二分类层，目测是用来match两个embedding的
        self.final = ArcMarginProduct(
            in_features,
            n_classes,
            scale=scale,
            margin=margin,
            easy_margin=False,
            ls_eps=0.0
        )

    def _init_params(self):
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    # model(input)默认调用该函数
    # 注意看这里，如果已经练好，直接就用features了（也就是embedding的结果）
    def forward(self, image, label):
        features = self.extract_features(image)
        if self.training:
            logits = self.final(features, label)
            return logits
        else:
            return features

    # 在forward函数中，通过model(input)进行调用
    def extract_features(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc and self.training:
            x = self.dropout(x)
            x = self.classifier(x)
            x = self.bn(x)
        return x


# ？处理图片描述信息
def get_test_transforms():
    return albumentations.Compose([
        albumentations.Resize(CFG.img_size, CFG.img_size, always_apply=True),
        albumentations.Normalize(),
        ToTensorV2(p=1.0)
    ])


# image embedding 流程方法
def get_image_embeddings(image_paths):

    model = ShopeeModel(pretrained=False).to(CFG.device)
    #
    model.load_state_dict(torch.load(CFG.model_path))
    model.eval()

    image_dataset = ShopeeDataset(image_paths=image_paths, transforms=get_test_transforms())
    image_loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers
    )

    embeds = []
    with torch.no_grad():
        for img,label in tqdm(image_loader):
            img = img.cuda()
            label = label.cuda()
            features = model(img,label)
            image_embeddings = features.detach().cpu().numpy()
            embeds.append(image_embeddings)

    image_embeddings = np.concatenate(embeds)
    print(f'Our image embeddings shape is {image_embeddings.shape}')
    return image_embeddings


# 把embedding后的图形进行KNN匹配
# KNN一般用来做分类，但是但是但是这里是借用了KNN的方法，来计算某个image最近的topk个image是哪些。
def get_image_neighbors(df, embeddings, KNN=100, threshold=4.5, metric='minkowski'):
    if metric == 'cosine':
        model = NearestNeighbors(n_neighbors=KNN, metric=metric, n_jobs=8)
    else:
        model = NearestNeighbors(n_neighbors=KNN, n_jobs=8)
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)

    predictions = []
    for k in tqdm(range(embeddings.shape[0])):
        idx = np.where(distances[k,] < threshold)[0]
        ids = indices[k, idx]
        posting_ids = df['posting_id'].iloc[ids].values
        predictions.append(posting_ids)

    return df, predictions