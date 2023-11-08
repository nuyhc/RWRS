import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
import warnings
warnings.filterwarnings("ignore")

color_map = {f"Trial {k+1}":v for k, v in zip([x for x in range(0, 10)], sns.color_palette())}
#
data = pd.read_csv("./Data/Prep_UsedCarSales/train.csv")
ftrs = data.columns.tolist()[:-1]
target = "Price"
#
x_train, x_val, y_train, y_val = train_test_split(data[ftrs], data[target], test_size=0.2, random_state=42)
train = pd.concat([x_train, y_train], axis=1).reset_index(drop=True)
valid = pd.concat([x_val, y_val], axis=1).reset_index(drop=True)

#########
data_n = 1
#########

# 모델 훈련
class ModelTrain:
    @staticmethod
    def statics(model, sample, valid_data=valid, ftrs=ftrs, target=target):
        model.fit(sample[ftrs], sample[target])
        return model.predict(valid_data[ftrs])
    @staticmethod
    def kmeans(model, sample, valid_data=valid, ftrs=ftrs, target=target):
        model.fit(sample[ftrs])
        db_result = pd.concat([sample.reset_index(drop=True), pd.DataFrame({"Train_Cluster":model.labels_})], axis=1)
        pred_result = pd.concat([valid_data.reset_index(drop=True), pd.DataFrame({"Pred_Cluster":model.predict(valid_data[ftrs])})], axis=1)
        db_result.columns = [x+"_DB" for x in ftrs]+[target, "Train_Cluster"]
        db_result = db_result.rename(columns={target:"Pred_Y"})
        pred_result.columns = [x+"_Pred" for x in ftrs]+[target, "Pred_Cluster"]
        pred_result = pred_result.rename(columns={target:"Act_Y"})
        #
        pred_train = pd.merge(
            left=pred_result, right=db_result,
            left_on="Pred_Cluster", right_on="Train_Cluster", how="left"
        )
        # 코사인 거리 & 유사도
        cd, cs = [], []
        for ridx in range(len(pred_train)):
            r_data = pred_train.iloc[ridx]
            db_ftrs = r_data[[x+"_DB" for x in ftrs]]
            input_ftrs = r_data[[x+"_Pred" for x in ftrs]]
            cd.append(cosine_distances(pd.DataFrame(db_ftrs).T.values, pd.DataFrame(input_ftrs).T.values)[0][0])
            cs.append(cosine_similarity(pd.DataFrame(db_ftrs).T.values, pd.DataFrame(input_ftrs).T.values)[0][0])
        pred_train["CosDist"] = cd
        pred_train["CosSim"] = cs
        return pred_train[["Pred_Y", "Act_Y", "CosDist", "CosSim"]]
    
def Train_N(n):
    sample_data = []
    pred_lrs, pred_dts, pred_rfs, pred_kmeans = [], [], [], []
    for _ in range(data_n):
        sample = data.sample(n, random_state=42)
        sample = sample.drop_duplicates()
        sample_data.append(sample)
        print(sample.shape[0], end=" ", sep=" ")
        #
        pred_lrs.append(ModelTrain.statics(LinearRegression(), sample))
        pred_dts.append(ModelTrain.statics(DecisionTreeRegressor(random_state=42, max_depth=3), sample))
        pred_rfs.append(ModelTrain.statics(RandomForestRegressor(random_state=42, max_depth=3), sample))
        pred_kmeans.append(ModelTrain.kmeans(KMeans(n_clusters=n, init="k-means++", random_state=42), sample))
    # 샘플링 결과 시각화
    for _ in range(len(sample_data)):
        sns.kdeplot(sample_data[_][target], fill=True, color=color_map[f"Trial {_+1}"])
    plt.legend(color_map.keys())
    plt.title(f"Sampling {n}")
    plt.show()
    return {
        "SamplingData":sample_data,
        "LR":pred_lrs,
        "DT":pred_dts,
        "RF":pred_rfs,
        "Kmeans":pred_kmeans
    }
    
def Train_N1(n):
    sample_data = []
    pred_lrs, pred_dts, pred_rfs, pred_kmeans = [], [], [], []
    #
    sample = data.sample(n, random_state=42)
    sample = sample.drop_duplicates()
    sample_data.append(sample)
    print(sample.shape[0], end=" ", sep=" ")
    #
    pred_lrs.append(ModelTrain.statics(LinearRegression(), sample))
    pred_dts.append(ModelTrain.statics(DecisionTreeRegressor(random_state=42, max_depth=3), sample))
    pred_rfs.append(ModelTrain.statics(RandomForestRegressor(random_state=42, max_depth=3), sample))
    pred_kmeans.append(ModelTrain.kmeans(KMeans(n_clusters=n, init="k-means++", random_state=42), sample))
    # 샘플링 결과 시각화
    for _ in range(len(sample_data)):
        sns.kdeplot(sample_data[_][target], fill=True, color=color_map[f"Trial {_+1}"])
    plt.legend(color_map.keys())
    plt.title(f"Sampling {n}")
    plt.show()
    return {
        "SamplingData":sample_data,
        "LR":pred_lrs,
        "DT":pred_dts,
        "RF":pred_rfs,
        "Kmeans":pred_kmeans
    }
    
#
def find_mean_mape(pred, valid_data=valid, target=target):
    p = pd.DataFrame(pred)
    best_mape, best_idx = 1, 0
    mapes = []
    for idx in range(len(p)):
        mape = mean_absolute_percentage_error(p.iloc[idx], valid_data[target])
        mapes.append(mape)
        if best_mape>mape: best_mape=mape
    return np.mean(mapes), best_mape

#
def plot_statics(result, n, valid_data=valid, ftrs=ftrs, target=target):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 7))
    #
    ax[0].plot(valid_data[target].values, color="black", linestyle=":")
    ax[0].plot(pd.DataFrame(result["LR"]).mean(axis=0), color="r")
    lr_m, lr_b = find_mean_mape(result["LR"])
    ax[0].set_title(f"LR\n{lr_m:.2f} / {lr_b:.2f}")
    #
    ax[1].plot(valid_data[target].values, color="black", linestyle=":")
    ax[1].plot(pd.DataFrame(result["DT"]).mean(axis=0), color="r")
    dt_m, dt_b = find_mean_mape(result["DT"])
    ax[1].set_title(f"DT\n{dt_m:.2f} / {dt_b:.2f}")
    #
    ax[2].plot(valid_data[target].values, color="black", linestyle=":")
    ax[2].plot(pd.DataFrame(result["RF"]).mean(axis=0), color="r")
    rf_m, rf_b = find_mean_mape(result["RF"])
    ax[2].set_title(f"RF\n{rf_m:.2f} / {rf_b:.2f}")
    plt.show()

def plot_kmeans(result, valid_data=valid, target=target):
    best_mape = 1
    mapes = []
    for idx in range(len(result["Kmeans"])):
        mape = mean_absolute_percentage_error(result["Kmeans"][idx]["Pred_Y"], valid_data[target].values)
        mapes.append(mape)
        if best_mape>mape: best_mape=mape
        #
        # prev_mape = best_mape      
        # for cor in np.arange(0.1, 1.5, 0.1):
        #     pprev_mape = prev_mape
        #     adj_mape=mean_absolute_percentage_error(result["Kmeans"][idx]["Pred_Y"]*cor, valid_data[target].values)
        #     if pprev_mape>adj_mape:
        #         if prev_mape>adj_mape:
        #             adj_mape = prev_mape
        #             adj_cor = cor
        #             select_sample = idx
        adj_cor = 0.8
        select_sample = -1
        adj_mape = mean_absolute_percentage_error(result["Kmeans"][idx]["Pred_Y"]*adj_cor, valid_data[target].values)

    r_m_mapes, r_b_mape = np.mean(mapes), best_mape
    #
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30, 7))
    #
    ax[0].plot(valid_data[target].values, color="black", linestyle=":")
    ax[0].plot(result["Kmeans"][select_sample]["Pred_Y"], color="r")
    ax[0].set_title(f"Kmeans\n{r_m_mapes:.2f} / {r_b_mape:.2f}")
    #
    ax[1].plot(valid_data[target].values, color="black", linestyle=":")
    ax[1].plot(result["Kmeans"][select_sample]["Pred_Y"]*adj_cor, color="r")
    ax[1].set_title(f"Kmeans\n{adj_mape:.2f} / {adj_cor:.2f} / in {select_sample}")
    plt.show()
    
def Exp(n):
    result = Train_N(n)
    plot_statics(result, n)
    plot_kmeans(result)
    return result

def train_once(n):
    sample = data.sample(n, random_state=42)
    sample.drop_duplicates()
    print(sample.shape[0])
    #
    lr = LinearRegression()
    pred_lr = ModelTrain.statics(lr, sample)
    dt = DecisionTreeRegressor(max_depth=3)
    pred_dt = ModelTrain.statics(dt, sample)
    rf = RandomForestRegressor(max_depth=3)
    pred_rf = ModelTrain.statics(rf, sample)
    kmeans = KMeans(n_clusters=10, init="k-means++")
    pred_k = ModelTrain.kmeans(kmeans, sample)
    #
    mape_lr = np.mean((abs(valid[target] - pred_lr))/valid[target])
    mape_dt = np.mean((abs(valid[target] - pred_dt))/valid[target])
    mape_rf = np.mean((abs(valid[target] - pred_rf))/valid[target])
    mape_k = np.mean(abs(valid[target] - pred_k["Pred_Y"])/valid[target])
    #
    best_mape = mape_k
    best_c = -1
    for c in np.arange(0.01 , 1.5, 0.01):
        mape_c = np.mean(abs(valid[target] - pred_k["Pred_Y"]*c)/valid[target])
        if mape_c < best_mape:
            best_mape = mape_c
            best_c = c
    # print(f"{mape_lr:.2f} {mape_dt:.2f} {mape_rf:.2f} {mape_k:.2f} {best_mape:.2f} {best_c}")
    #
    # fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(30, 5))
    # ax[0].plot(valid[target].values, color="black", linestyle=":")
    # ax[0].plot(pred_lr, color="red")
    # ax[0].set_title(f"LR: {mape_lr:.2f}")
    # #
    # ax[1].plot(valid[target].values, color="black", linestyle=":")
    # ax[1].plot(pred_dt, color="red")
    # ax[1].set_title(f"DT: {mape_dt:.2f}")
    # #
    # ax[2].plot(valid[target].values, color="black", linestyle=":")
    # ax[2].plot(pred_rf, color="red")
    # ax[2].set_title(f"RF: {mape_rf:.2f}")
    # #
    # ax[3].plot(valid[target].values, color="black", linestyle=":")
    # ax[3].plot(pred_k["Pred_Y"], color="red")
    # ax[3].set_title(f"Kmeans: {mape_k:.2f}")
    # #
    # ax[4].plot(valid[target].values, color="black", linestyle=":")
    # ax[4].plot(pred_k["Pred_Y"]*best_c, color="red")
    # ax[4].set_title(f"Kmeans*{best_c}:{best_mape:.2f}")
    return pd.DataFrame({
        "Sample#":[n],
        "LR":mape_lr,
        "DT":mape_dt,
        "RF":mape_rf,
        "Kmeans":mape_k,
        "adj_Kmeans":best_mape,
        "adj_corr":[best_c]
    })