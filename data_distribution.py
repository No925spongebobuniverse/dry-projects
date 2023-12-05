import pandas as pd

"""
    将数据集按6：2：2分为训练集和测试集和验证集
"""

if __name__ == "__main__":
    data = pd.read_csv("D:/cleaned_data.csv", low_memory=False,header=None)  #加载数据
    data:pd.DataFrame = data.sample(frac=1.0)                     #将数据打乱
    rows, cols = data.shape
    split_index_1 = int(rows * 0.2)
    split_index_2 = int(rows * 0.4)
    #数据分割
    data_test:pd.DataFrame = data.iloc[0: split_index_1, :]
    data_validate:pd.DataFrame = data.iloc[split_index_1:split_index_2, :]
    data_train:pd.DataFrame = data.iloc[split_index_2: rows, :]
    #数据保存
    data_test.to_csv("cleaned_data-test.csv", header=None, index=False)
    data_validate.to_csv("cleaned_data-validate.csv", header=None, index=False)
    data_train.to_csv("cleaned_data-train.csv", header=None, index=False)
    print("Distribution finished.")

