import pandas as pd
from dataAugmentation import sqrtData
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import scipy
import warnings
import hydroeval as he
import time

warnings.filterwarnings("ignore")


class featureCorrelation():
    def __init__(self, file, model=None, modelName='', shuffle=False):
        self.epoch = 0

        ## Prepare the data
        self.df = pd.read_csv(file).drop('sort', axis=1).dropna()
        no3 = np.asarray(self.df["no3"])
        no3 = np.exp(no3) - 1
        no3 = sqrtData(sqrtData(no3))
        self.df["no3"] = no3

        self.dataLength = len(self.df)
        self.model = model
        self.modelName = modelName
        self.regime = 'train'
        self.runSignature = 'train'
        self.FEATURES = ['spec_power_1.5', 'spec_power_3.0', 'spec_power_6.0', 'spec_power_12.0', 'spec_power_24.0',
                         'spec_power_48.0', 'spec_power_96.0', 'spec_power_192.0', 'spec_power_384.0',
                         'area', 'elev', 'forest', 'wetland', 'urban', 'ag', 'roads', 'normalRandom']
        self.TARGETS = ["doc", "no3", "tn", "tp", "random"]

        ## Create random test ouputs
        self.df["random"] = np.random.uniform(0, 1, len(self.df[self.df.columns[0]]))

        ## Create random test inputs
        self.df["normalRandom"] = np.random.normal(0, 1, (len(self.df[self.df.columns[0]])))

        ## Create output dictionarys
        self.dataDict = {"testIndex": [], "modeltype": [], 'trainRegime': [], "doc": [], "no3": [], "tn": [], "tp": [],
                         "random": [], "runSig": [], "epoch": []}
        self.featureImportances = {"testIndex": [], "modeltype": [], "targetNutrient": [], 'trainRegime': [], "epoch": []}
        self.resultsDict = {"modeltype": [], 'trainRegime': [], "doc": [], "no3": [], "tn": [], "tp": [],
                            "random": [], "epoch": [], "evaluation_criterion": []}
        for feature in self.FEATURES:
            self.featureImportances[feature] = []
        self.curFeatures = self.FEATURES
        self.X = self.df[self.FEATURES].to_numpy()

    # def updateModel(self, model, modelName, features='all', runSignature='train', epoch=0):
    #     self.model = model
    #     self.modelName = modelName
    #     self.regime = features
    #     self.runSignature = runSignature
    #     self.epoch = epoch
    #
    #     if features == 'Flow':
    #         ## Shuffle the catchment features
    #         shuffleFeatures = self.FEATURES[10:17]
    #         df = self.df.copy()
    #         for feature in shuffleFeatures:
    #             df[feature] = np.random.permutation(df[feature])
    #         self.X = df[self.FEATURES].to_numpy()
    #
    #     if features == 'Catchment':
    #         ## Shuffle the flow features
    #         shuffleFeatures = self.FEATURES[:9]
    #         df = self.df.copy()
    #         for feature in shuffleFeatures:
    #             df[feature] = np.random.permutation(df[feature])
    #         self.X = df[self.FEATURES].to_numpy()
    #
    #     if features == 'random':
    #         shuffleFeatures = self.FEATURES
    #         df = self.df.copy()
    #         for feature in shuffleFeatures:
    #             df[feature] = np.random.permutation(df[feature])
    #         self.X = df[self.FEATURES].to_numpy()
    #     else:
    #         ## include everything
    #         self.X = self.df[self.FEATURES].to_numpy()

        def updateModel(self, model, modelName, epoch, features='all', runSignature='train', ):
            self.model = model
            self.modelName = modelName
            self.regime = features
            self.runSignature = runSignature
            self.epoch = epoch

            if features == 'Flow':
                ## Shuffle the catchment features
                shuffleFeatures = self.FEATURES[9:]
                df = self.df.copy()
                for feature in shuffleFeatures:
                    df[feature] = np.random.permutation(df[feature])
                self.X = df[self.FEATURES].to_numpy()


            elif features == 'Catchment':
                ## Shuffle the flow features
                shuffleFeatures = self.FEATURES[:9]
                df = self.df.copy()
                for feature in shuffleFeatures:
                    df[feature] = np.random.permutation(df[feature])
                self.X = df[self.FEATURES].to_numpy()
            elif features == 'Random':
                ## random data
                self.X = np.random.random(self.df[self.FEATURES].shape)

            elif features == 'all':
                ## include everything
                self.X = self.df[self.FEATURES].to_numpy()

            else:
                print('Not a valid train resime')

    def resetResults(self):
        self.dataDict = {"testIndex": [], "modeltype": [], 'trainRegime': [], "doc": [], "no3": [], "tn": [], "tp": [],
                         "random": [], "runSig": [], "epoch": []}
        self.featureImportances = {"testIndex": [], "modeltype": [], "targetNutrient": [], 'trainRegime': [], "epoch": []}
        self.resultsDict = {"modeltype": [], 'trainRegime': [], "doc": [], "no3": [], "tn": [], "tp": [],
                            "random": [], "epoch": [], "evaluation_criterion": []}
        for feature in self.FEATURES:
            self.featureImportances[feature] = []

    def train(self):
        assert self.model != None, 'No model found'
        ### the 5 is for testing faster
        for i in range(self.dataLength):
            self.dataDict["testIndex"].append(i)
            self.dataDict['trainRegime'].append(self.regime)
            self.dataDict["modeltype"].append(self.modelName)
            self.dataDict['runSig'].append(self.runSignature)
            self.dataDict['epoch'].append(self.epoch)
            trainX = np.delete(self.X, i, axis=0)

            ## Make predictions for each of the 5 chemicals in rivers
            for targetNutrient in self.TARGETS:
                Y = self.df[targetNutrient].to_numpy()
                trainY = np.delete(Y, i, axis=0)
                self.model.fit(trainX, trainY)

                y_hat = self.model.predict(self.X[i].reshape(-1, len(self.curFeatures)))
                y_truth = Y[i].reshape(-1, 1)

                self.dataDict[targetNutrient].append(np.array([y_hat, y_truth], dtype=object))

                if self.modelName in ['GBR', 'RFR', 'DTR']:
                    self.featureImportances["testIndex"].append(i)
                    self.featureImportances["modeltype"].append(self.modelName)
                    self.featureImportances["targetNutrient"].append(targetNutrient)
                    self.featureImportances['trainRegime'].append(self.regime)
                    self.featureImportances['epoch'].append(self.epoch)
                    importances = self.model.feature_importances_
                    for j, feature in enumerate(self.FEATURES):
                        self.featureImportances[feature].append(importances[j])

        ## appends the NSE scores to results dictionary

        for evalType in ["rmse","mare","nse","pbias","rsquared"]:
            self.resultsDict['modeltype'].append(self.modelName)
            self.resultsDict['trainRegime'].append(self.regime)
            self.resultsDict['epoch'].append(self.epoch)
            self.resultsDict['evaluation_criterion'].append(evalType)

            for nutrient in ["doc", "no3", "tn", "tp", "random"]:
                data = np.array(self.dataDict[nutrient])
                y_hats = data[:, 0].squeeze()
                y_truth = data[:, 1].squeeze()

                if evalType == "rmse":
                    acc = he.evaluator(he.rmse, list(y_hats), list(y_truth))[0]
                elif evalType == "mare":
                    acc = he.evaluator(he.mare, list(y_hats), list(y_truth))[0]
                elif evalType == "nse":
                    acc = he.evaluator(he.nse, list(y_hats), list(y_truth))[0]
                elif evalType == "pbias":
                    acc = he.evaluator(he.pbias, list(y_hats), list(y_truth))[0]
                elif evalType == "r-squared":
                    slope, intercept, acc, p_value, std_err = scipy.stats.linregress(list(y_hats), list(y_truth))

                self.resultsDict[nutrient].append(acc)

        return self.resultsDict

    def exportResults(self, predictionsFileName, featureImportancesFileName, accuracyFileName):
        #
        dataDF = pd.DataFrame.from_dict(self.dataDict)
        for nutrient in ["doc", "no3", "tn", "tp", "random"]:
            data = np.array(self.dataDict[nutrient])
            dataDF[nutrient] = data[:, 0]
            dataDF[nutrient + "_true"] = data[:, 1]

        dataDF.to_csv(predictionsFileName, index=False)
        pd.DataFrame.from_dict(self.featureImportances).to_csv(featureImportancesFileName, index=False)
        pd.DataFrame.from_dict(self.resultsDict).to_csv(accuracyFileName, index=False)

    def plotResultsViolin(self, title, model='all'):
        df = pd.DataFrame(self.resultsDict)

        ## chose which model data to plot
        if model != 'all':
            df = df[df['modeltype'] == model]

        ## plot violin plots for each of the 5 outputs
        for nutrient in ["doc", "no3", "tn", "tp", "random"]:
            nu_df = df[df["evaluation_criterion"] == "r-squared"]
            nu_df = nu_df[[nutrient, 'trainRegime', 'modeltype']]

            nu_df = nu_df.rename(columns={nutrient: 'accuracy'})

            # nu_df['accuracy'] = nu_df['accuracy'].apply(lambda x: (x[1][0] - x[0][0]) ** 2) #this is for training one epoch and using data dict
            sns.violinplot(data=nu_df, x="trainRegime", y="accuracy", hue="modeltype")
            plt.title(f"{nutrient} Feature {title} Accuracies")
            plt.ylabel("coefficient of determinination")
            plt.xlabel("training regime")
            plt.show()

    def plotResults(self, title, runSignature='all'):
        df = pd.DataFrame(self.dataDict)
        ## chose which run to plot
        if runSignature != 'all':
            df = df[df['runSig'] == runSignature]

        ## scatter plot of predicted vs true results
        for nutrient in ["doc", "no3", "tn", "tp", "random"]:
            data = np.hstack(df[nutrient].to_numpy()).T
            y_hats = data[:, 0]
            y_truth = data[:, 1]
            # plt.scatter(y_truth, y_hats, label=nutrient, alpha=0.7)
        # plt.plot([0, 1], [0, 1], label='line of perfect score')
        # plt.title(title)
        # plt.xlabel('True value')
        # plt.ylabel('Predicted value')
        # plt.legend()
        # plt.show()


def hyperParmetersTuner(epochs, datafile, hyperparams, model, modelname):
    start_time = time.time()
    FC = featureCorrelation(datafile, model(), modelName=modelname)
    scores = []
    times = []

    for param in hyperparams:
        for epoch in range(epochs):
            if modelname == 'GBR' or modelname == 'RFR':
                curModel = model(n_estimators=param)
            if modelname == 'KNN':
                curModel = model(n_neighbors=param)
            if modelname == 'MLP':
                curModel = model(max_iter=param)
            FC.updateModel(curModel, modelName=f'{modelname}-{param}', epoch=epoch)
            results = FC.train()

        score = np.mean([np.mean(results[type]) for type in ["doc", "no3", "tn", "tp"]])
        print(f'{modelname}-{param} : {score}')
        scores.append(score)
        FC.resetResults()

        times.append(time.time() - start_time)
        start_time = time.time()

    return hyperparams[np.argmax(scores)], scores, times


## run this to get final test results
def getResults(models, modelNames, dataFile, epochs=1, exportFiles=['', '']):
    dpi = 100
    sns.set(rc={"figure.dpi": dpi, 'savefig.dpi': dpi})
    plt.rcParams['figure.dpi'] = dpi
    FC = featureCorrelation(dataFile)
    for model, modelName in zip(models, modelNames):
        for resime in ['allFeatures', 'Flow', 'Catchment', 'Random']:
            for epoch in range(epochs):
                FC.updateModel(model, modelName, resime, runSignature=f'{modelName}-{resime}', epoch=epoch)
                ## runs model 800 times to get a distribution for single predictions
                FC.train()
            FC.plotResults(f'{modelName}-{resime}', f'{modelName}-{resime}')
            FC.exportResults(exportFiles[0], exportFiles[1], exportFiles[2])
        print(f'finished training {modelName}')
        FC.plotResultsViolin(f'{modelName}', f'{modelName}')
    FC.plotResultsViolin(f'{modelName}', 'all')
    FC.exportResults(exportFiles[0], exportFiles[1], exportFiles[2])

## run this to find the best parameters
def tuneHyperParameters(file):
    estimators = [1, 5, 10, 20, 40, 80, 100, 200, 400, 600]
    estimators = [1, 2]
    bestGBR, scoresGBR, timesGBR = hyperParmetersTuner(3, file, estimators, GradientBoostingRegressor, 'GBR')
    bestMLP, scoresMLP, timesMLP = hyperParmetersTuner(3, file, estimators, MLPRegressor, 'MLP')
    bestRFR, scoresRFR, timesRFR = hyperParmetersTuner(3, file, estimators, RandomForestRegressor, 'RFR')

    knnEstimators = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    bestKNN, scoresKNN, timesKNN = hyperParmetersTuner(3, file, knnEstimators, KNeighborsRegressor, 'KNN')

    print('\nHyperparameter Tuning Results')
    print(f'bestGBR: {bestGBR} \nscoresGBR: {scoresGBR} \ntimesGBR: {timesGBR}\n')
    print(f'bestMLP: {bestMLP} \nscoresMLP: {scoresMLP} \ntimesMLP: {timesMLP}\n')
    print(f'bestKNN: {bestKNN} \nscoresKNN: {scoresKNN} \ntimesKNN: {timesKNN}\n')
    print(f'bestRFR: {bestRFR} \nscoresRFR: {scoresRFR} \ntimesRFR: {timesRFR}\n')


if __name__ == '__main__':
    file = "../data/epa_minmax_complete_TRAIN.csv"
    '''
    gbr n_estimators = 100
    rfr n_estimators = 100
    knn: k=6 
    '''

    models = [GradientBoostingRegressor(n_estimators=100), KNeighborsRegressor(n_neighbors=6),
              DecisionTreeRegressor(), RandomForestRegressor(n_estimators=100), MLPRegressor()]
    modelNames = ['GBR', 'KNN', 'DTR', 'RFR', 'MLP']
    getResults(models, modelNames, file, epochs=1,
               exportFiles=['predictions.csv', 'featureImportances.csv', 'accuracyResults.csv'])