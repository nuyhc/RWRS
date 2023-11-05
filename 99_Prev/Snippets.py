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
data_size = 5

# Load
color_map = {f"Trial {k}":v for k, v in zip([x for x in range(0, 10)], sns.color_palette())}
#
train = pd.read_csv("./Data/Prep_AMES/train.csv")
ftrs = train.columns.tolist()[:-1]
x_train, x_val, y_train, y_val = train_test_split(train[ftrs], train["Y"], test_size=0.1, random_state=42)

# 모델 훈련
class ModelTrain:
    @staticmethod
    def statics(model: Model, sample_data=pd.DataFrame)->list:
        model.fit(sample_data[ftrs], sample_data["Y"])
        return model.predict(x_val)
    @staticmethod
    def kmeans(model: Model, sample_data=pd.DataFrame)->list:
        model.fit(sample_data[ftrs])
        train_result = pd.concat([sample_data.reset_index(drop=True), pd.DataFrame({"Train_Cluster":model.labels_})], axis=1)
        pred_result = pd.concat([x_val.reset_index(drop=True), y_val.reset_index(drop=True), pd.DataFrame({"Pred_Cluster":model.predict(x_val.values)})], axis=1)
        train_result.columns = [x+"_t" for x in ftrs]+["Pred_Y", "Train_Cluster"]
        pred_result.columns = [x+"_p" for x in ftrs]+["Act_Y", "Pred_Cluster"]
        #
        pred_train = pd.merge(
            left=pred_result, right=train_result,
            left_on="Pred_Cluster", right_on="Train_Cluster"
        )
        # 코사인 거리 & 유사도
        cd, cs = [], []
        for idx in range(len(pred_train)):
            r_data = pred_train.iloc[idx]
            recommand_ftrs = r_data[[x+"_t" for x in ftrs]]
            input_ftrs = r_data[[x+"_p" for x in ftrs]]
            cd.append(cosine_distances(pd.DataFrame(recommand_ftrs).T, pd.DataFrame(input_ftrs).T)[0][0])
            cs.append(cosine_similarity(pd.DataFrame(recommand_ftrs).T, pd.DataFrame(input_ftrs).T)[0][0])
        pred_train["CosDist"] = cd
        pred_train["CosSim"] = cs
        pred_train["GrLivArea P/T"] = pred_train["GrLivArea_p"] / pred_train["GrLivArea_t"]
        pred_train["GrLivArea T/P"] = pred_train["GrLivArea_t"] / pred_train["GrLivArea_p"]
        return pred_train[["Pred_Y", "Act_Y", "CosDist", "CosSim", "GrLivArea P/T", "GrLivArea T/P"]]
    
# 결과 시각화
class Plot:
    @staticmethod
    def statics(pred: list, model_name: str):
        r2s, mapes = [], []
        for idx in range(data_size):
            r2s.append(r2_score(pred[idx], y_val.values))
            mapes.append(mean_absolute_percentage_error(pred[idx], y_val.values))
        sns.scatterplot(x=r2s, y=mapes)
        plt.xlabel("R2")
        plt.ylabel("MAPE")
        plt.axvline(0, color="r", linestyle=":")
        plt.axhline(0, color="r", linestyle=":")
        plt.title(f"[{model_name}]\nBest R2: {max(r2s):.2f} / MAPE: {(min(mapes)):.2f}\nMedian R2: {np.median(r2s):.2f} / MAPE: {np.median(mapes):.2f}\nMean R2:{np.mean(r2s):.2f} / MAPE: {np.mean(mapes):.2f}")
        plt.show()
    @staticmethod
    def kmeans(pred: list, model_name: str):
        r2s, mapes = [], []
        r2s_cCS, mapes_cCS = [], []
        r2s_cGPT, mapes_cGPT = [], []
        r2s_cCD, mapes_cCD = [], []
        for idx in range(data_size):
            r2s.append(r2_score(pred[idx]["Pred_Y"], y_val.values))
            mapes.append(mean_absolute_percentage_error(pred[idx]["Pred_Y"], y_val.values))
            # 코사인 유사도 보정
            r2s_cCS.append(r2_score(pred[idx]["Pred_Y"]*(2-pred[idx]["CosSim"]), y_val.values))
            mapes_cCS.append(mean_absolute_percentage_error(pred[idx]["Pred_Y"]*(2-pred[idx]["CosSim"]), y_val.values))
            # GrLivArea P/T 보정
            r2s_cGPT.append(r2_score(pred[idx]["Pred_Y"]*pred[idx]["GrLivArea P/T"], y_val.values))
            mapes_cGPT.append(mean_absolute_percentage_error(pred[idx]["Pred_Y"]*pred[idx]["GrLivArea P/T"], y_val.values))
            # 코사인 거리 보정
            r2s_cCD.append(r2_score(pred[idx]["Pred_Y"]*(2-pred[idx]["CosDist"]), y_val.values))
            mapes_cCD.append(mean_absolute_percentage_error(pred[idx]["Pred_Y"]*(2-pred[idx]["CosDist"]), y_val.values))
        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
        #
        sns.scatterplot(x=r2s, y=mapes, ax=ax[0])
        ax[0].set_xlabel("R2")
        ax[0].set_ylabel("MAPE")
        ax[0].axvline(0, color="r", linestyle=":")
        ax[0].axhline(0, color="r", linestyle=":")
        ax[0].set_title(f"[KMeans]\nBest R2: {max(r2s):.2f} / MAPE: {(min(mapes)):.2f}\nMedian R2: {np.median(r2s):.2f} / MAPE: {np.median(mapes):.2f}\nMean R2:{np.mean(r2s):.2f} / MAPE: {np.mean(mapes):.2f}")
        #
        sns.scatterplot(x=r2s_cCS, y=mapes_cCS, ax=ax[1])
        ax[1].set_xlabel("R2")
        ax[1].set_ylabel("MAPE")
        ax[1].axvline(0, color="r", linestyle=":")
        ax[1].axhline(0, color="r", linestyle=":")
        ax[1].set_title(f"[KMeans(CS)]\nBest R2: {max(r2s_cCS):.2f} / MAPE: {(min(mapes_cCS)):.2f}\nMedian R2: {np.median(r2s_cCS):.2f} / MAPE: {np.median(mapes_cCS):.2f}\nMean R2:{np.mean(r2s_cCS):.2f} / MAPE: {np.mean(mapes_cCS):.2f}")
        #
        sns.scatterplot(x=r2s_cGPT, y=mapes_cGPT, ax=ax[2])
        ax[2].set_xlabel("R2")
        ax[2].set_ylabel("MAPE")
        ax[2].axvline(0, color="r", linestyle=":")
        ax[2].axhline(0, color="r", linestyle=":")
        ax[2].set_title(f"[KMeans(GP/T)]\nBest R2: {max(r2s_cGPT):.2f} / MAPE: {(min(mapes_cGPT)):.2f}\nMedian R2: {np.median(r2s_cGPT):.2f} / MAPE: {np.median(mapes_cGPT):.2f}\nMean R2:{np.mean(r2s_cGPT):.2f} / MAPE: {np.mean(mapes_cGPT):.2f}")
        #
        sns.scatterplot(x=r2s_cCD, y=mapes_cCD, ax=ax[3])
        ax[3].set_xlabel("R2")
        ax[3].set_ylabel("MAPE")
        ax[3].axvline(0, color="r", linestyle=":")
        ax[3].axhline(0, color="r", linestyle=":")
        ax[3].set_title(f"[KMeans(CD)]\nBest R2: {max(r2s_cCD):.2f} / MAPE: {(min(mapes_cCD)):.2f}\nMedian R2: {np.median(r2s_cCD):.2f} / MAPE: {np.median(mapes_cCD):.2f}\nMean R2:{np.mean(r2s_cCD):.2f} / MAPE: {np.mean(mapes_cCD):.2f}")
        plt.show()

# (종합)모델 훈련 및 예측
def train_model(n: int)->dict:
    sample_data = []
    pred_lrs, pred_dts, pred_rfs, pred_kmeans = [], [], [], []
    model_lrs, model_dts, model_rfs, model_kmeans = [], [], [], []
    for idx in range(data_size):
        sample = train.sample(n)
        sample = sample.drop_duplicates()
        sample_data.append(sample)
        print(sample.shape[0], end=" ", sep= " ")
        #
        lr = LinearRegression()
        dt = DecisionTreeRegressor(random_state=42)
        rf = RandomForestRegressor(random_state=42)
        km = KMeans(n_clusters=n, init="k-means++")
        # 모델 훈련 및 예측 결과
        pred_lrs.append(ModelTrain.statics(lr, sample))
        pred_dts.append(ModelTrain.statics(dt, sample))
        pred_rfs.append(ModelTrain.statics(rf, sample))
        pred_kmeans.append(ModelTrain.kmeans(km, sample))
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
    
# (종합)모델 훈련 및 예측
def train_model_skew(n: int)->dict:
    sample_data = []
    pred_lrs, pred_dts, pred_rfs, pred_kmeans = [], [], [], []
    model_lrs, model_dts, model_rfs, model_kmeans = [], [], [], []
    mu = np.mean(train["Y"])
    std = np.std(train["Y"])
    skew_train = pd.concat([train[train["Y"]>=mu+std], train[train["Y"]<=mu-std]], axis=0).reset_index(drop=True)
    for idx in range(data_size):
        sample = skew_train.sample(n)
        sample = sample.drop_duplicates()
        sample_data.append(sample)
        print(sample.shape[0], end=" ", sep= " ")
        #
        lr = LinearRegression()
        dt = DecisionTreeRegressor(random_state=42)
        rf = RandomForestRegressor(random_state=42)
        km = KMeans(n_clusters=n, init="k-means++")
        # 모델 훈련 및 예측 결과
        pred_lrs.append(ModelTrain.statics(lr, sample))
        pred_dts.append(ModelTrain.statics(dt, sample))
        pred_rfs.append(ModelTrain.statics(rf, sample))
        pred_kmeans.append(ModelTrain.kmeans(km, sample))
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
