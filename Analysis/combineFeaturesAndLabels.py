import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def logData(data):
    values = np.array(data)
    values = values + 1
    values = np.log(values)
    return values

def sqrtData(data):
    values = np.array(data)
    values = values + 0.0000000001
    values = np.sqrt(values)
    return values


dfFeatures = pd.read_csv("../data/epa_mean_powers.csv")
dfTargets = pd.read_csv("../data/epa_stream_data.csv")
dfFeaturesT = dfFeatures.T
newCols =  list(dfFeatures.index)
print(newCols)
print(dfFeaturesT)
dfFeaturesT.columns = dfFeaturesT.iloc[0]
featureCols = ["spec_power_" + str(x) for x in dfFeaturesT.columns]
dfFeaturesT.columns = featureCols
dfFeaturesT = dfFeaturesT.drop("Unnamed: 0", axis=0)
df = dfFeaturesT

# add sort column
sorts = df.index
df["sort"] = sorts
df = df.drop(df.columns[0], axis=1)
df["sort"] = [str(x) for x in df["sort"]]
dfTargets["sort"] = [str(x) for x in dfTargets["sort"]]

# merge dataframes
df = df.merge(dfTargets, on="sort")
extraFeatures = ["area","elev","forest","wetland","urban","ag","roads"]
keeperCols = featureCols[1:] + extraFeatures + ["doc","no3","tn","tp"]
sorts = df["sort"]
df = df[keeperCols]


# mergedMeanStd = df.merge(dfTargets, on="sort")

# normalize
# df = dfFeaturesT
print(df)
# df = (df - df.mean()) / (df.std())
# df = (df - df.min()) / (df.max() - df.min())

for col in df.columns:
    df[col] = logData(df[col])
df = (df - df.min()) / (df.max() - df.min())
df["sort"] = sorts


print(df)
print(df.describe())

# quit()
# df = pd.read_csv("epa_combined_minmax_minimal.csv")
# df["doc"] = logData(df["doc"])
# df["no3"] = sqrtData(sqrtData(df["no3"]))
# df["tn"] = logData(df["tn"])
# df["tp"] = logData(df["tp"])
# df[targets] = (df[targets] - df[targets].min()) / (df[targets].max() - df[targets].min())
#
# df.to_csv("epa_combined_minmax_minimal_normalizedTargets.csv", index=False)
#
# df = pd.read_csv("epa_combined_meanstd_minimal.csv")
# print(df)
# df["doc"] = logData(df["doc"])
# df["no3"] = sqrtData(sqrtData(df["no3"]))
# df["tn"] = logData(df["tn"])
# df["tp"] = logData(df["tp"])
# df[targets] = (df[targets] - df[targets].mean()) / (df[targets].std())
# # df[targets] = (df[targets] - df[targets].min()) / (df[targets].max() - df[targets].min())
# df.to_csv("epa_combined_meanstd_minimal_normalizedTargets.csv", index=False)
#


# mergedMeanStd = mergedMeanStd[keeperCols]

import random
X = df.to_numpy()
indices = list(range(X.shape[0]))
random.shuffle(indices)
trainIndices = indices[:800]
testIndices = indices[800:]

dfTrain = df.iloc[trainIndices]
dfTest = df.iloc[testIndices]

# mergedMeanStdTrain = mergedMeanStd.iloc[trainIndices]
# mergedMeanStdTest = mergedMeanStd.iloc[testIndices]

dfTrain.to_csv("epa_minmax_complete_TRAIN.csv", index=False)
# mergedMeanStdTrain.to_csv("epa_combined_meanstd_complete_TRAIN.csv", index=False)
dfTest.to_csv("epa_minmax_complete_TEST.csv", index=False)
# mergedMeanStdTest.to_csv("epa_combined_meanstd_complete_TEST.csv", index=False)
# for col in dfTrain.columns:
#     plt.hist(dfTrain[col], bins=40, density=True, alpha=0.5)
#     plt.hist(dfTest[col], bins=40, density=True, alpha=0.5)
#     plt.title(col)
#     plt.show()
#

keeperCols = featureCols[1:]+ ["doc","no3","tn","tp"]

dfTrainMin = dfTest[keeperCols]
dfTestMin = dfTrain[keeperCols]

dfTrainMin.to_csv("epa_minmax_minimal_TRAIN.csv", index=False)
dfTestMin.to_csv("epa_minmax_minimal_TEST.csv", index=False)



# mergedMeanStdTrain = mergedMeanStdTrain[keeperCols]
# mergedMeanStdTest = mergedMeanStdTest[keeperCols]

# mergedMeanStdTrain.to_csv("epa_combined_meanstd_minimal_TRAIN.csv", index=False)
# mergedMeanStdTest.to_csv("epa_combined_meanstd_minimal_TEST.csv", index=False)
