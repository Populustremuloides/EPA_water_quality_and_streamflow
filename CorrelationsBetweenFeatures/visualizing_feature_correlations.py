import pandas as pd
from matplotlib import pyplot as plt

def resultsPlot(df, type, drop_columns):
    df = df[df['modeltype'] == type].copy()
    df = df.drop(columns=drop_columns)
    for col in df.columns:
        plt.hist(df[col], label=col)
    plt.title(type)
    plt.show()
def featureImportantePlot(df, type, drop_columns):
    df = df[df['modeltype'] == type]
    df = df.drop(columns=drop_columns)
    for t in ['doc','no3','tn','tp','random']:
        df_target = df[df['target']==t].copy()

        for col in df_target.columns:
            if col =='target':
                continue
            plt.hist(df_target[col], label=col)
        plt.legend()
        plt.title(f'{type}: {t}')
        plt.show()
def plotFeatureCorrelation(csv, plot='results', drop_columns=['repeat', 'modeltype']):
    dataFrame = pd.read_csv(csv)
    if plot == 'results':
        ## Random forest regressor plot
        resultsPlot(dataFrame, 'RFR', drop_columns)
        ## Gradient boosted regressor plot
        resultsPlot(dataFrame, 'GBR', drop_columns)
    else:
        featureImportantePlot(dataFrame, 'RFR', drop_columns)
        ## Gradient boosted regressor plot
        featureImportantePlot(dataFrame, 'GBR', drop_columns)

if __name__ == '__main__':
    plotFeatureCorrelation('minmax_prediction_results_randomX.csv')
    #plotFeatureCorrelation('minmax_prediction_featureImportances_randomX.csv',plot='feature_importance')
