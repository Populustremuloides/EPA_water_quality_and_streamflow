import pandas as pd
# import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dfDTR = pd.read_csv("../MixMaxResults/minmax_prediction_results_DTR.csv")
dfGBR = pd.read_csv("../MixMaxResults/minmax_prediction_results_GBR.csv")
dfKNN = pd.read_csv("../MixMaxResults/minmax_prediction_results_KNN.csv")
dfMLP = pd.read_csv("../MixMaxResults/minmax_prediction_results_MLP.csv")
dfRFR = pd.read_csv("../MixMaxResults/minmax_prediction_results_RFR.csv")

total = dfDTR.append(dfGBR)
total = total.append(dfKNN)
total = total.append(dfMLP)
total = total.append(dfRFR)

dataDict = {"modelType":[],"nutrient":[],"accuracy":[]}
nutrients = ["doc","no3","tn","tp","random"]

for index, row in total.iterrows():
    for nutrient in nutrients:
        dataDict["nutrient"].append(nutrient)
        dataDict["modelType"].append(row["modeltype"])
        dataDict["accuracy"].append(row[nutrient])

df = pd.DataFrame.from_dict(dataDict)

sns.catplot(data=df, x="modelType",y="accuracy",hue="nutrient")
plt.title("Initial Test of Different Models - Default Hyperparameters")
plt.ylim(-0.5,0.5)
plt.show()

print(total)
