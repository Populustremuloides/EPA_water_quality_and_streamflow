import pandas as pd
# import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dfGBR = pd.read_csv("../MixMaxResults/minmax_prediction_results_GBR_nestimators.csv")
dfKNN = pd.read_csv("../MixMaxResults/minmax_prediction_results_KNN_numNeighbors.csv")
dfRFR = pd.read_csv("../MixMaxResults/minmax_prediction_results_RFR_nestimators.csv")

total = dfGBR.append(dfRFR)
# total = total.append(dfKNN)

dataDict = {"modelType":[],"nutrient":[],"accuracy":[],"number of estimators":[]}
nutrients = ["doc","no3","tn","tp"]


for index, row in total.iterrows():
    for nutrient in nutrients:
        dataDict["nutrient"].append(nutrient)
        dataDict["modelType"].append(row["modeltype"])
        dataDict["accuracy"].append(row[nutrient])
        dataDict["number of estimators"].append(row["n_estimators"])

df = pd.DataFrame.from_dict(dataDict)



rfrdf = df[df["modelType"] == "RFR"]
sns.lineplot(data=rfrdf, x="number of estimators",y="accuracy",hue="nutrient")
plt.title("Tuning Hyperparameters - Random Forest")
plt.xscale("log")
plt.ylim(0,0.63)
plt.show()


gbrdf = df[df["modelType"] == "GBR"]
sns.lineplot(data=gbrdf, x="number of estimators",y="accuracy",hue="nutrient")
plt.title("Tuning Hyperparameters Gradient Boosting")
plt.xscale("log")
plt.ylim(0,0.63)
plt.show()

dataDict2 = {"modelType":[],"nutrient":[],"accuracy":[],"k":[]}
for index, row in dfKNN.iterrows():
    for nutrient in nutrients:
        dataDict2["nutrient"].append(nutrient)
        dataDict2["modelType"].append(row["modeltype"])
        dataDict2["accuracy"].append(row[nutrient])
        dataDict2["k"].append(row["n_estimators"])
kdf = pd.DataFrame.from_dict(dataDict2)
sns.lineplot(data=kdf, x="k",y="accuracy",hue="nutrient")
plt.title("Tuning Hyperparameters K-Nearest Neighbors")
plt.ylim(0,0.63)
plt.show()