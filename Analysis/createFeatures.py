import numpy as np
import pandas as pd

df = pd.read_csv("../data/epa_spectral_data.csv")
print(df)

def getPowersOfTwo(pow):
    powers = [0]
    for i in range(pow):
        powers.append(2 ** i)
    return powers

newData = {}
for col in df.columns:
    newData[col] = []

X = df.to_numpy()
indices = getPowersOfTwo(10)

outIndices = []
for i in range(len(indices) - 1):
    startIndex = indices[i]
    endIndex = indices[i + 1]
    outIndices.append(np.mean([startIndex, endIndex]))

for col in df.columns:
    newData[col] = []
    for i in range(len(indices) - 1):
        startIndex = indices[i]
        endIndex = indices[i + 1]
        meanPower = np.mean(df[col][startIndex:endIndex])
        # print(meanPower)
        # print(type(meanPower))
        if np.isnan(meanPower):
            print(df[col][startIndex:endIndex])
        # if meanPower == np.na:
        #     print(df[col][startIndex:endIndex])
        # print(meanPower)
        newData[col].append(meanPower)

    # print(X[startIndex:endIndex,:])
    # meanPowers = np.nanmean(X[startIndex:endIndex,:], axis=0)
    # print(meanPowers.shape)
    #
    # for j, col in enumerate(df.columns):
    #     newData[col].append(meanPowers[j]) # check the indices here
outDf = pd.DataFrame.from_dict(newData)
outDf.index = outIndices
    # print(outDf)
# outDf.index = indices
outDf.to_csv("epa_mean_powers.csv")