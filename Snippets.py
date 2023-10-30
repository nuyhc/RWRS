import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from scipy import stats
from scipy.stats import norm, skew
import warnings
warnings.filterwarnings("ignore")

from typing import TypeVar

Model = TypeVar("Model")

# Load
color_map = {f"Trial {k}":v for k, v in zip([x for x in range(0, 10)], sns.color_palette())}
#
train = pd.read_csv("./Data/Prep_AMES/train.csv")
ftrs = train.columns.tolist()[:-1]
x_train, x_val, y_train, y_val = train_test_split(train[ftrs], train["Y"], test_size=0.2, random_state=327)

def cal_cosin_distance(cols): return cosine_distances(cols[train_ftrs], cols[pred_ftrs])
def cal_cosin_similarity(cols): return cosine_similarity(cols[train_ftrs], cols[pred_ftrs])

# 모델 훈련
class ModelTrain:
    @staticmethod
    def statics(model: Model, sample_data=pd.DataFrame)->list:
        model.fit(sample_data[ftrs], sample_data["Y"])
        return model.predict(x_val)
    @staticmethod
    def kmeans(model: Model, sample_data=pd.DataFrame)->list:
        pass
    
# 결과 시각화
class Plot:
    @staticmethod
    def statics(pred: list, model_name: str):
        r2s, mapes = [], []
        for idx in range(10):
            r2s.append(r2_score(pred[idx], y_val.values))
            mapes.append(mean_absolute_percentage_error(pred[idx], y_val.values))
        sns.scatterplot(x=r2s, y=mapes)
        plt.xlabel("R2")
        plt.ylabel("MAPE")
        plt.axvline(0, color="r", linestyle=":")
        plt.axhline(0, color="r", linestyle=":")
        plt.title(f"[{model_name}]\nBest R2: {max(r2s):.2f} / MAPE: {(min(mapes)):.2f}\nMedian R2: {np.median(r2s):.2f} / MAPE: {np.median(mapes):.2f}\nMean R2:{np.mean(r2s):.2f} / MAPE: {np.mean(mapes):.2f}")
        plt.show()

# (종합)모델 훈련 및 예측
def train_model(n: int)->dict:
    sample_data = []
    pred_lrs, pred_dts, pred_rfs, pred_kmeans = [], [], [], []
    model_lrs, model_dts, model_rfs, model_kmeans = [], [], [], []
    for idx in range(10):
        sample = train.sample(n)
        sample = sample.drop_duplicates()
        sample_data.append(sample)
        #
        lr = LinearRegression()
        dt = DecisionTreeRegressor(random_state=42)
        rf = RandomForestRegressor(random_state=42)
        km = KMeans()
        # 모델 훈련 및 예측 결과
        pred_lrs.append(ModelTrain.statics(lr, sample))
        pred_dts.append(ModelTrain.statics(dt, sample))
        pred_rfs.append(ModelTrain.statics(rf, sample))
        pred_kmeans.append(Model)
    # 샘플링 데이터 시각화
    for idx in range(len(sample_data)): sns.kdeplot(sample_data[idx]["Y"], fill=True, color=color_map[f"Trial {idx}"])
    plt.legend(color_map.keys())
    plt.title(f"Sampling {n}")
    plt.show()
    # 모델 성능 평가
    
    return {
        "SampleData":sample_data,
        "LR":pred_lrs,
        "DT":pred_dts,
        "RF":pred_rfs,
        "Kmeans":pred_kmeans
    }