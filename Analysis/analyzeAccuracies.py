import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df1 = pd.read_csv("../MixMaxResults/minmax_prediction_results.csv")
df2 = pd.read_csv("../MixMaxResults/minmax_prediction_results_complete.csv")
df3 = pd.read_csv("../MixMaxResults/minmax_prediction_results_flowonly.csv")
df4 = pd.read_csv("../MixMaxResults/minmax_prediction_results_randomX.csv")

dataDict = {"nutrient":[],"accuracy":[],"trainingRegime":[],"repeat":[], "model type":[]}

def updateDataDict(dataDict, df, description):

    for index, row in df.iterrows():
        repeat = row["repeat"]
        modelType = row["modeltype"]
        trainingRegime = description

        for nutrient in ["doc","no3","tn","tp","random"]:
            accuracy = row[nutrient]

            dataDict["nutrient"].append(nutrient)
            dataDict["accuracy"].append(accuracy)
            dataDict["trainingRegime"].append(trainingRegime)
            dataDict["repeat"].append(repeat)
            dataDict["model type"].append(modelType)

    return dataDict

dataDict = updateDataDict(dataDict, df1, "catchmentChars")
dataDict = updateDataDict(dataDict, df2, "full")
dataDict = updateDataDict(dataDict, df3, "flowRegime")
dataDict = updateDataDict(dataDict, df4, "random")

df = pd.DataFrame.from_dict(dataDict)
df.to_csv("minmax_accuracies_longformat.csv", index=False)

docDf = df[df["nutrient"] == "doc"]

sns.violinplot(data=docDf, x="trainingRegime",y="accuracy",hue="model type")
plt.title("DOC Model Accuracies, n=100 each")
plt.ylabel("coefficient of determinination")
plt.xlabel("training regime")
plt.show()
print(docDf)

no3Df = df[df["nutrient"] == "no3"]
sns.violinplot(data=no3Df, x="trainingRegime",y="accuracy",hue="model type")
plt.title("NO3 Model Accuracies, n=100 each")
plt.ylabel("coefficient of determinination")
plt.xlabel("training regime")
plt.show()

tnDf = df[df["nutrient"] == "tn"]
sns.violinplot(data=tnDf, x="trainingRegime",y="accuracy",hue="model type")
plt.title("Total Nitrogen Model Accuracies, n=100 each")
plt.ylabel("coefficient of determinination")
plt.xlabel("training regime")
plt.show()

tpDf = df[df["nutrient"] == "tp"]
sns.violinplot(data=tpDf, x="trainingRegime",y="accuracy",hue="model type")
plt.title("Total Phosphorus Model Accuracies, n=100 each")
plt.ylabel("coefficient of determinination")
plt.xlabel("training regime")
plt.show()

tpDf = df[df["nutrient"] == "random"]

sns.violinplot(data=tpDf, x="trainingRegime",y="accuracy",hue="model type")
plt.title("Random Feature Model Accuracies, n=100 each")
plt.ylabel("coefficient of determinination")
plt.xlabel("training regime")
plt.show()

# for nutrient in ["doc","no3","tn","tp","random"]:
#     ldf1 = df1[df1["nutrient"] == nutrient]
#     ldf2 = df2[df2["nutrient"] == nutrient]
#     ldf3 = df3[df2["nutrient"] == nutrient]
#     ldf4 = df4[df2["nutrient"] == nutrient]
#
#     for index, row in ldf1.iterrows():
