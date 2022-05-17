import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

dfComplete = pd.read_csv("../MixMaxResults/minmax_prediction_featureImportances_complete.csv")
dfPartial = pd.read_csv("../MixMaxResults/minmax_prediction_featureImportances.csv")
print(dfComplete)
print(dfPartial)

dataDict = {"nutrient":[],"importance":[],"repeat":[],"complete":[]}

def updateDataDict(dataDict, row, col, complete):
    dataDict["nutrient"].append(col)
    dataDict["importance"].append(row[col])
    dataDict["repeat"].append(row["repeat"])
    dataDict["complete"].append(complete)
    print(row["repeat"])
    if row["repeat"] > 100:
        dataDict["modeltype"].append("GBR")
    else:
        dataDict["modeltype"].append("RFR")
    return dataDict

for i, col in enumerate(dfComplete.columns):
    print(i, col)
for i, col in enumerate(dfPartial.columns):
    print(i, col)

# print(list(set(dfPartial.columns).difference(set(dfComplete))))
# quit()
#
# dfComplete = dfComplete[dfComplete["repeat"] > 99]
# keeperCols = dfComplete.columns[3:]
# for nutrient in ["doc","no3","tn","tp"]:
#     dataDict = {"nutrient": [], "importance": [], "repeat": [], "complete": [], "modeltype":[]}
#     ldf = dfComplete[dfComplete["target"] == nutrient]
#     for index, row in ldf.iterrows():
#         for col in keeperCols:
#             dataDict = updateDataDict(dataDict, row, col, "complete")
#
#     df = pd.DataFrame.from_dict(dataDict)
#     print(dataDict)
#     sns.violinplot(data=df, x="nutrient", y="importance")
#     plt.title(nutrient)
#     plt.xticks(rotation=45)
#     plt.show()


# dfPartial = dfPartial[dfPartial["repeat"] > 99]
keeperCols = dfComplete.columns[3:]
for nutrient in ["doc","no3","tn","tp"]:
    dataDict = {"nutrient": [], "importance": [], "repeat": [], "complete": [], "modeltype":[]}
    ldf = dfComplete[dfComplete["target"] == nutrient]
    for index, row in ldf.iterrows():
        for col in keeperCols:
            dataDict = updateDataDict(dataDict, row, col, "partial")
    df = pd.DataFrame.from_dict(dataDict)
    # print(dataDict)
    sns.violinplot(data=df, x="nutrient", y="importance", hue="modeltype")
    plt.title(nutrient + " model feature importances")
    # plt.hlines(y=1, xmin=0, xmax=20) # np.mean(df["importance"][df["nutrient"] == "randomNormal"])
    plt.xticks(rotation=90)
    plt.show()

quit()

#
# df = pd.DataFrame.from_dict(dataDict)
# sns.violinplot(data=df, x="nutrient", y="accuracy", hue="complete")
# plt.show()


print(dfComplete)
print(dfPartial)
quit()




dfComplete = pd.read_csv("../MixMaxResults/minmax_prediction_results_complete.csv")
dfPartial = pd.read_csv("../MixMaxResults/minmax_prediction_results.csv")

print(dfComplete.describe())
print(dfPartial.describe())

dataDict = {"nutrient":[],"accuracy":[],"repeat":[],"complete":[]}
for index, row in dfComplete.iterrows():
    dataDict["nutrient"].append("doc")
    dataDict["accuracy"].append(row["doc"])
    dataDict["repeat"].append(row["repeat"])
    dataDict["complete"].append("with_flow")

    dataDict["nutrient"].append("no3")
    dataDict["accuracy"].append(row["no3"])
    dataDict["repeat"].append(row["repeat"])
    dataDict["complete"].append("with_flow")

    dataDict["nutrient"].append("tn")
    dataDict["accuracy"].append(row["tn"])
    dataDict["repeat"].append(row["repeat"])
    dataDict["complete"].append("with_flow")

    dataDict["nutrient"].append("tp")
    dataDict["accuracy"].append(row["tp"])
    dataDict["repeat"].append(row["repeat"])
    dataDict["complete"].append("with_flow")

for index, row in dfComplete.iterrows():
    dataDict["nutrient"].append("doc")
    dataDict["accuracy"].append(row["doc"])
    dataDict["repeat"].append(row["repeat"])
    dataDict["complete"].append("without_flow")

    dataDict["nutrient"].append("no3")
    dataDict["accuracy"].append(row["no3"])
    dataDict["repeat"].append(row["repeat"])
    dataDict["complete"].append("without_flow")

    dataDict["nutrient"].append("tn")
    dataDict["accuracy"].append(row["tn"])
    dataDict["repeat"].append(row["repeat"])
    dataDict["complete"].append("without_flow")

    dataDict["nutrient"].append("tp")
    dataDict["accuracy"].append(row["tp"])
    dataDict["repeat"].append(row["repeat"])
    dataDict["complete"].append("without_flow")

df = pd.DataFrame.from_dict(dataDict)
sns.violinplot(data=df, x="nutrient",y="accuracy",hue="complete")
plt.show()


