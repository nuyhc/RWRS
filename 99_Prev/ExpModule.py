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
from sklearn.feature_selection import RFE
#
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
#
import warnings
warnings.filterwarnings("ignore")
color_map = {f"Trial {k}":v for k, v in zip([x for x in range(0, 10)], sns.color_palette())}
#
data = pd.read_csv("./Data/Prep_AMES/sig_train.csv").iloc[:, 1:]
data["Label"] = data["Y"].apply(lambda x: str(int(x/100000)))
#
ftrs = data.columns.tolist()[:-2]
# x_train, x_val, y_train, y_val = train_test_split(data[ftrs], data["Y"], test_size=0.05, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(data[ftrs], data["Y"], stratify=data["Label"], test_size=0.05, random_state=42)
train = pd.concat([x_train, y_train], axis=1).reset_index(drop=True)
valid = pd.concat([x_val, y_val], axis=1).reset_index(drop=True)
#
data_n = 5
#
# RFE 이용해, n개의 피처 선택
def select_n_ftrs(model, n, data=train, ftrs=ftrs, target="Y"):
    selector = RFE(model, n_features_to_select=n, step=1)
    selector = selector.fit(data[ftrs], data[target])
    sel = []
    for val, selected in zip(ftrs, selector.support_):
        if selected: sel.append(val)
    return sel

# 모델 훈련
class ModelTrain:
    @staticmethod
    def statics(model, sample_data, selected_ftrs=ftrs, target="Y"):
        model.fit(sample_data[selected_ftrs], sample_data[target])
        return model.predict(x_val[selected_ftrs])
    @staticmethod
    def kmeans(model, sample_data, selected_ftrs=ftrs):
        model.fit(sample_data[selected_ftrs])
        train_result = pd.concat([sample_data[selected_ftrs].reset_index(drop=True), sample_data["Y"].reset_index(drop=True), pd.DataFrame({"Train_Cluster":model.labels_})], axis=1)
        pred_result = pd.concat([x_val[selected_ftrs].reset_index(drop=True), y_val.reset_index(drop=True), pd.DataFrame({"Pred_Cluster":model.predict(x_val[selected_ftrs].values)})], axis=1)
        #
        train_result.columns = [x+"_Train" for x in selected_ftrs]+["Y", "Train_Cluster"]
        train_result = train_result.rename(columns={"Y":"Pred_Y"})
        pred_result.columns = [x+"_Pred" for x in selected_ftrs]+["Y", "Pred_Cluster"]
        pred_result = pred_result.rename(columns={"Y":"Act_Y"})
        #
        pred_train = pd.merge(
            left=pred_result, right=train_result,
            left_on="Pred_Cluster", right_on="Train_Cluster"
        )
        # 코사인 거리 & 유사도
        cd, cs = [], []
        for idx in range(len(pred_train)):
            r_data = pred_train.iloc[idx]
            recommand_ftrs = r_data[[x+"_Train" for x in selected_ftrs]]
            input_ftrs = r_data[[x+"_Pred" for x in selected_ftrs]]
            cd.append(cosine_distances(pd.DataFrame(recommand_ftrs).T, pd.DataFrame(input_ftrs).T)[0][0])
            cs.append(cosine_similarity(pd.DataFrame(recommand_ftrs).T, pd.DataFrame(input_ftrs).T)[0][0])
        pred_train["CosDist"] = cd
        pred_train["CosSim"] = cs
        pred_train["GrLivArea P/T"] = pred_train["GrLivArea_Pred"] / pred_train["GrLivArea_Train"]
        pred_train["GrLivArea T/P"] = pred_train["GrLivArea_Train"] / pred_train["GrLivArea_Pred"]
        return pred_train[["Pred_Y", "Act_Y", "CosDist", "CosSim", "GrLivArea P/T", "GrLivArea T/P"]]

# n개 훈련
def train_n(n, sn, data=train, target="Y"):
    sample_data,  pred_lrs, pred_dts, pred_rfs, pred_kmeans = [], [], [], [], []
    sftrs_lrs, sftrs_dts, sftrs_rfs, sftrs_kmeans = [], [], [], []
    for _ in range(data_n):
        sample = data.sample(n)
        sample = sample.drop_duplicates()
        sample_data.append(sample)
        print(sample.shape[0], end=" ", sep=" ")
        #
        lr = LinearRegression()
        dt = DecisionTreeRegressor(random_state=42, max_depth=4)
        rf = RandomForestRegressor(random_state=42, max_depth=4)
        km = KMeans(n_clusters=n, init="k-means++")
        #
        # sftr_lr = select_n_ftrs(lr, sn)
        # sftr_dt = select_n_ftrs(dt, sn)
        # sftr_rf = select_n_ftrs(rf, sn)
        #
        # statics_selected = pd.DataFrame({"Selected":sftr_lr+sftr_dt+sftr_rf})
        # statics_selected = pd.DataFrame(statics_selected.groupby("Selected")["Selected"].count())
        # statics_selected = statics_selected.rename_axis("Ftrs").reset_index(drop=False).sort_values("Selected", ascending=False)
        # sftr_k = statics_selected["Ftrs"].tolist()[:sn]
        #
        pred_lrs.append(ModelTrain.statics(lr, sample))
        pred_dts.append(ModelTrain.statics(dt, sample))
        pred_rfs.append(ModelTrain.statics(rf, sample))
        pred_kmeans.append(ModelTrain.kmeans(km, sample))
        # pred_lrs.append(ModelTrain.statics(lr, sample, sftr_lr))
        # pred_dts.append(ModelTrain.statics(dt, sample, sftr_dt))
        # pred_rfs.append(ModelTrain.statics(rf, sample, sftr_rf))
        # pred_kmeans.append(ModelTrain.kmeans(km, sample, sftr_k))
        #
        # sftrs_lrs.append(sftr_lr)
        # sftrs_dts.append(sftr_dt)
        # sftrs_rfs.append(sftr_rf)
        # sftrs_kmeans.append(sftr_k)
    # 샘플링 데이터 시각화
    for _ in range(len(sample_data)): sns.kdeplot(sample_data[_]["Y"], fill=True, color=color_map[f"Trial {_}"])
    plt.legend(color_map.keys())
    plt.title(f"Sampling {n}")
    plt.show()
    #
    return {
        "SampleData":sample_data,
        "LR":{"P":pred_lrs, "F":sftrs_lrs},
        "DT":{"P":pred_dts, "F":sftrs_dts},
        "RF":{"P":pred_rfs, "F":sftrs_rfs},
        "Kmeans":{"P":pred_kmeans, "F":sftrs_kmeans}
    }
    
def plot_static(result, n, sn):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 7))
    #
    ax[0].plot(y_val.values, color="black", linestyle=":") # 원본
    ax[0].plot(pd.DataFrame(result["LR"]["P"]).mean(axis=0).values, color="r")
    r2 = r2_score(y_val.values, pd.DataFrame(result["LR"]["P"]).mean(axis=0).values)
    mape = (abs(y_val.values - pd.DataFrame(result["LR"]["P"]).mean(axis=0).values) / y_val.values).mean()
    # mape = mean_absolute_percentage_error(y_val.values, pd.DataFrame(result["LR"]["P"]).mean(axis=0).values)
    ax[0].set_title(f"LR\nM.R2: {r2:.4f}\nM.MAPE: {mape:.4f}")
    #
    ax[1].plot(y_val.values, color="black", linestyle=":") # 원본
    ax[1].plot(pd.DataFrame(result["DT"]["P"]).mean(axis=0).values, color="r")
    r2 = r2_score(y_val.values, pd.DataFrame(result["DT"]["P"]).mean(axis=0).values)
    # mape = mean_absolute_percentage_error(y_val.values, pd.DataFrame(result["DT"]["P"]).mean(axis=0).values)
    mape = (abs(y_val.values - pd.DataFrame(result["DT"]["P"]).mean(axis=0).values) / y_val.values).mean()
    ax[1].set_title(f"DT\nM.R2: {r2:.4f}\nM.MAPE: {mape:.4f}")
    #
    ax[2].plot(y_val.values, color="black", linestyle=":") # 원본
    ax[2].plot(pd.DataFrame(result["RF"]["P"]).mean(axis=0).values, color="r")
    r2 = r2_score(y_val.values, pd.DataFrame(result["RF"]["P"]).mean(axis=0).values)
    # mape = mean_absolute_percentage_error(y_val.values, pd.DataFrame(result["RF"]["P"]).mean(axis=0).values)
    mape = (abs(y_val.values - pd.DataFrame(result["RF"]["P"]).mean(axis=0).values) / y_val.values).mean()
    ax[2].set_title(f"RF\nM.R2: {r2:.4f}\nM.MAPE: {mape:.4f}")
    #
    plt.suptitle(f"Sample {n} / Ftrs {sn}")
    
def plot_kmeans(result, c=0.8):
    pred_y, cor_cd, cor_cs, cor_gpt, cor_gtp = [], [], [], [], []
    # best C
    # cors = []
    for idx in range(data_n):
        # c = (result["Kmeans"]["P"][idx]["Pred_Y"]/result["Kmeans"]["P"][idx]["Act_Y"]).mean()
        # print(c, end=" ", sep=" ")
        prev_mape = 1
        print(f"보정전: {c}")
        for cor in np.arange(0.1, 3, 0.01):
            if prev_mape > (abs(y_val.values - result["Kmeans"]["P"][0]["Pred_Y"]*cor)/y_val.values).mean():
                prev_mape = (abs(y_val.values - result["Kmeans"]["P"][idx]["Pred_Y"]*cor) / y_val.values).mean()
                c = cor
        # print(f"보정치: {c}")
        # cors.append(c)
        pred_y.append(result["Kmeans"]["P"][idx]["Pred_Y"]*c)
        cor_cd.append(result["Kmeans"]["P"][idx]["Pred_Y"]*(1-result["Kmeans"]["P"][idx]["CosDist"])*c)
        cor_cs.append(result["Kmeans"]["P"][idx]["Pred_Y"]*(result["Kmeans"]["P"][idx]["CosSim"])*c)
        cor_gpt.append(result["Kmeans"]["P"][idx]["Pred_Y"]*result["Kmeans"]["P"][idx]["GrLivArea P/T"]*c)
        cor_gtp.append(result["Kmeans"]["P"][idx]["Pred_Y"]*result["Kmeans"]["P"][idx]["GrLivArea T/P"]*c)
    #
    pred_y = pd.DataFrame(pred_y)
    cor_cd = pd.DataFrame(cor_cd)
    cor_cs = pd.DataFrame(cor_cs)
    cor_gpt = pd.DataFrame(cor_gpt)
    cor_gtp = pd.DataFrame(cor_gtp)
    #
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(30, 5))
    #
    ax[0].plot(y_val.values, color="black", linestyle=":")
    ax[0].plot(pred_y.mean(axis=0), color="r")
    r2 = r2_score(y_val.values, pred_y.mean(axis=0))
    # mape = mean_absolute_percentage_error(y_val, pred_y.mean(axis=0))
    mape = (abs(y_val.values - pred_y.mean(axis=0)) / y_val.values).mean()
    ax[0].set_title(f"Kmeans\nM.R2: {r2:.4f}\nM.MAPE: {mape:.4f}")
    #
    ax[1].plot(y_val.values, color="black", linestyle=":")
    ax[1].plot(cor_cd.mean(axis=0), color="r")
    r2 = r2_score(y_val.values, cor_cd.mean(axis=0))
    # mape = mean_absolute_percentage_error(y_val, cor_cd.mean(axis=0))
    mape = (abs(y_val.values - cor_cd.mean(axis=0)) / y_val.values).mean()
    ax[1].set_title(f"Kmeans(CD)\nM.R2: {r2:.4f}\nM.MAPE: {mape:.4f}")
    #
    ax[2].plot(y_val.values, color="black", linestyle=":")
    ax[2].plot(cor_cs.mean(axis=0), color="r")
    r2 = r2_score(y_val.values, cor_cs.mean(axis=0))
    # mape = mean_absolute_percentage_error(y_val, cor_cs.mean(axis=0))
    mape = (abs(y_val.values - cor_cs.mean(axis=0)) / y_val.values).mean()
    ax[2].set_title(f"Kmeans(CS)\nM.R2: {r2:.4f}\nM.MAPE: {mape:.4f}")
    #
    ax[3].plot(y_val.values, color="black", linestyle=":")
    ax[3].plot(cor_gpt.mean(axis=0), color="r")
    r2 = r2_score(y_val.values, cor_gpt.mean(axis=0))
    # mape = mean_absolute_percentage_error(y_val, cor_gpt.mean(axis=0))
    mape = (abs(y_val.values - cor_gpt.mean(axis=0)) / y_val.values).mean()
    ax[3].set_title(f"Kmeans(P/T)\nM.R2: {r2:.4f}\nM.MAPE: {mape:.4f}")
    #
    ax[4].plot(y_val.values, color="black", linestyle=":")
    ax[4].plot(cor_gtp.mean(axis=0), color="r")
    r2 = r2_score(y_val.values, cor_gtp.mean(axis=0))
    # mape = mean_absolute_percentage_error(y_val, cor_gtp.mean(axis=0))
    mape = (abs(y_val.values - cor_gtp.mean(axis=0)) / y_val.values).mean()
    ax[4].set_title(f"Kmeans(T/P)\nM.R2: {r2:.4f}\nM.MAPE: {mape:.4f}")
    #
    plt.show()
    return c
    
# 앙상블 시각화
def plot_ensemble(result, w, c=0.8):
    pred_y, cor_cd, cor_cs, cor_gpt, cor_gtp = [], [], [], [], []
    for idx in range(data_n):
        pred_y.append(result["Kmeans"]["P"][idx]["Pred_Y"]*c)
        cor_cd.append(result["Kmeans"]["P"][idx]["Pred_Y"]*(1-result["Kmeans"]["P"][idx]["CosDist"])*c)
        cor_cs.append(result["Kmeans"]["P"][idx]["Pred_Y"]*(result["Kmeans"]["P"][idx]["CosSim"])*c)
        cor_gpt.append(result["Kmeans"]["P"][idx]["Pred_Y"]*result["Kmeans"]["P"][idx]["GrLivArea P/T"]*c)
        cor_gtp.append(result["Kmeans"]["P"][idx]["Pred_Y"]*result["Kmeans"]["P"][idx]["GrLivArea T/P"]*c)
    #
    pred_y = pd.DataFrame(pred_y)
    cor_cd = pd.DataFrame(cor_cd)
    cor_cs = pd.DataFrame(cor_cs)
    cor_gpt = pd.DataFrame(cor_gpt)
    cor_gtp = pd.DataFrame(cor_gtp)
    #
    pred_lr_k = pd.DataFrame(result["LR"]["P"]).mean(axis=0)*(1-w) + pred_y.mean(axis=0)*w
    pred_dt_k = pd.DataFrame(result["DT"]["P"]).mean(axis=0)*(1-w) + pred_y.mean(axis=0)*w
    pred_rf_k = pd.DataFrame(result["RF"]["P"]).mean(axis=0)*(1-w) + pred_y.mean(axis=0)*w
    #
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 9))
    #
    ax[0].plot(y_val.values, color="black", linestyle=":")
    ax[0].plot(pred_lr_k, color="r")
    r2 = r2_score(y_val.values, pred_lr_k)
    # mape = mean_absolute_percentage_error(y_val.values, pred_lr_k)
    mape = (abs(y_val.values - pred_lr_k) / y_val.values).mean()
    ax[0].set_title(f"LR-K\nM.R2: {r2:.4f}\nM.MAPE: {mape:.4f}")
    #
    ax[1].plot(y_val.values, color="black", linestyle=":")
    ax[1].plot(pred_dt_k, color="r")
    r2 = r2_score(y_val.values, pred_dt_k)
    # mape = mean_absolute_percentage_error(y_val.values, pred_dt_k)
    mape = (abs(y_val.values - pred_dt_k) / y_val.values).mean()
    ax[1].set_title(f"DT-K\nM.R2: {r2:.4f}\nM.MAPE: {mape:.4f}")
    #
    ax[2].plot(y_val.values, color="black", linestyle=":")
    ax[2].plot(pred_rf_k, color="r")
    r2 = r2_score(y_val.values, pred_rf_k)
    # mape = mean_absolute_percentage_error(y_val.values, pred_rf_k)
    mape = (abs(y_val.values - pred_rf_k) / y_val.values).mean()
    ax[2].set_title(f"LR-K\nM.R2: {r2:.4f}\nM.MAPE: {mape:.4f}")
    #
    plt.suptitle(f"K weight={w}")
    plt.show()
    
# Train
def Train(sample_n):
    result = train_n(n=sample_n, sn=13)
    plot_static(result, n=sample_n, sn=13)
    c = plot_kmeans(result)
    c = int(c*100)/100
    print(f"C={c}")
    plot_ensemble(result, w=0.6, c=c)
    return result
        