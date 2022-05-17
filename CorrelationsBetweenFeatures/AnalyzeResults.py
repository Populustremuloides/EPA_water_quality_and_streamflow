import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df = pd.read_csv("predictions.csv")
print(df.columns)
for regime in ["allFeatures","Catchment","Flow","Random"]:
    ldf = df[df["trainRegime"] == regime]
    plt.scatter(ldf["doc"], ldf["doc_true"])
    plt.show()