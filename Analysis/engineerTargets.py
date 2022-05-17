import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

targets = ["doc","no3","tn","tp"]

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

df = pd.read_csv("../data/epa_combined_minmax_minimal.csv")
df["doc"] = logData(df["doc"])
df["no3"] = sqrtData(sqrtData(df["no3"]))
df["tn"] = logData(df["tn"])
df["tp"] = logData(df["tp"])
df[targets] = (df[targets] - df[targets].min()) / (df[targets].max() - df[targets].min())

df.to_csv("epa_combined_minmax_minimal_normalizedTargets.csv", index=False)

df = pd.read_csv("../data/epa_combined_meanstd_minimal.csv")
print(df)
df["doc"] = logData(df["doc"])
df["no3"] = sqrtData(sqrtData(df["no3"]))
df["tn"] = logData(df["tn"])
df["tp"] = logData(df["tp"])
df[targets] = (df[targets] - df[targets].mean()) / (df[targets].std())
# df[targets] = (df[targets] - df[targets].min()) / (df[targets].max() - df[targets].min())
df.to_csv("epa_combined_meanstd_minimal_normalizedTargets.csv", index=False)


#
# plt.hist(df["doc"], bins=40, label="original", density=True)
# plt.title("distribution of DOC")
# plt.xlabel("doc value")
# plt.ylabel("density")
# plt.show()
# doc = logData(df["doc"])
# plt.hist(doc, bins=40, label="transformed", density=True)
# plt.title("distribution of log(DOC)")
# plt.xlabel("log(doc) value")
# plt.ylabel("count")
# plt.show()
#
# plt.hist(df["no3"], bins=40, label="original", density=True)
# plt.title("distribution of NO3")
# plt.xlabel("no3 value")
# plt.ylabel("density")
# plt.show()
# no3 = sqrtData(sqrtData(df["no3"]))
# plt.hist(no3, bins=40, label="transformed", density=True)
# plt.title("distribution of (NO3)^(1/4)")
# plt.xlabel("(no3)^(1/4) value")
# plt.ylabel("count")
# plt.show()
#
# plt.hist(df["tn"], bins=40, label="original", density=True)
# plt.title("distribution of tn")
# plt.xlabel("tn value")
# plt.ylabel("density")
# plt.show()
# tn = logData(df["tn"])
# plt.hist(tn, bins=40, label="transformed", density=True)
# plt.title("distribution of log(tn)")
# plt.xlabel("log(tn) value")
# plt.ylabel("count")
# plt.show()
#
# plt.hist(df["tp"], bins=40, label="original", density=True)
# plt.title("distribution of tp")
# plt.xlabel("tp value")
# plt.ylabel("density")
# plt.show()
# tp = logData(df["tp"])
# plt.hist(tp, bins=40, label="transformed", density=True)
# plt.title("distribution of log(tp)")
# plt.xlabel("log(tp) value")
# plt.ylabel("count")
# plt.show()
#
#
