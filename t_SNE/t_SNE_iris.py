from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np

iris = load_iris()
df = pd.DataFrame(data = np.c_[iris.data, iris.target], 
                columns=["sepal length", "sepal width", "petal length", 
                        "petal width", "target"])



# class target 정보 제외
train_df = df[["sepal length", "sepal width", "petal length", "petal width"]]

# 2차원 t-SNE 임베딩
tsne_np = TSNE(n_components=2).fit_transform(train_df)

# numpy array -> DataFrame 변환
tsne_df = pd.DataFrame(tsne_np, columns = ["component 0", "component 1"])

import matplotlib.pyplot as plt

# class target 정보 불러오기
tsne_df["target"] = df["target"]

# target 별 분리
tsne_df_0 = tsne_df[tsne_df["target"] == 0]
tsne_df_1 = tsne_df[tsne_df["target"] == 1]
tsne_df_2 = tsne_df[tsne_df["target"] == 2]

# target 별 시각화
plt.scatter(tsne_df_0["component 0"], tsne_df_0["component 1"], color="pink",label="setosa")
plt.scatter(tsne_df_1["component 0"], tsne_df_1["component 1"], color="purple",label="versicolor")
plt.scatter(tsne_df_2["component 0"], tsne_df_2["component 1"], color="yellow",label="virginica")


plt.xlabel("component 0")
plt.ylabel("component 1")
plt.legend()
plt.show()













