import pandas as pd
import pyAgrum as gum
import numpy as np
import copy

def replaceNA(data, missMap):
    df = copy.deepcopy(data)
    for var in missMap.keys():
        values = df[var]
        ind = df[missMap[var]]
        nans = [pd.NA]*len(values)
        result = np.where(ind == 1, nans, values)
        df[var] = result
    return df

def predRow(rowDict, model, oracle=True, missingIndicator=True):
    # Predict a row (using evidence passed as a dictionary: rowDict).
    # - if Oracle is True, the X1 value will be kept event if M1 = 1
    # - if missingIndicator is True, M1 will be kept
    evidence = copy.deepcopy(rowDict)

    if oracle == False:
        if rowDict['M1'] == 1:
            evidence.pop("X1")

    if (missingIndicator == False):
        evidence.pop("M1")

    print("Evidence: ", evidence)
    ie = gum.LazyPropagation(model)
    ie.setEvidence(evidence)
    ie.makeInference()
    return ie.posterior("Y")[1]

structureStrings = {"S3_1": "X1{0|1}->X2{0|1}->Y{0|1};X1->Y; M1{0|1}; X3{0|1}",
                    "S3_2": "X1{0|1}->X2{0|1}->Y{0|1};X1->Y; X2->M1{0|1}; X3{0|1}",
                    "S3_3": "X1{0|1}->X2{0|1}->Y{0|1};X1->Y; X3{0|1}->M1{0|1};X3->X2",
                    "S3_4": "X1{0|1}->X2{0|1}->Y{0|1};X1->Y; X1->M1{0|1}; X3{0|1}",
                    "S3_5": "X1{0|1}->X2{0|1}->Y{0|1};X1->Y; Y->M1{0|1}; X3{0|1}",
                    "S3_6": "X1{0|1}->X2{0|1}->Y{0|1};X1->Y; Y->M1{0|1}<-X1; X3{0|1}"}

dSim = pd.read_csv("simulatedData.csv")

iteration = 0
structureLabel = "S3_5"

dataTrain = copy.deepcopy(dSim[(dSim["iteration"] == iteration) &
                               (dSim["structureLabel"] == structureLabel) &
                               (dSim["setType"] == "TRAIN")])
dataTrainNA = replaceNA(dataTrain, missMap = {"X1": "M1"})
print(dataTrainNA.head)
dataTest = copy.deepcopy(dSim[(dSim["iteration"] == iteration) &
                              (dSim["structureLabel"] == structureLabel) &
                              (dSim["setType"] == "TEST")])
dataTestNA = replaceNA(dataTest, missMap = {"X1": "M1"})
print(dataTestNA.head)

# -----------
#
# Predictions
#
# -----------

# ------- Marginalisation -------
# fit Bayesian Network
bnMissStruct = gum.fastBN(structureStrings[structureLabel])
learner = gum.BNLearner(dataTrainNA,bnMissStruct)
learner.useEM(1e-3)
learner.useSmoothingPrior()
learner.learnParameters(bnMissStruct.dag())

# predict in testSet
dictTest = dataTest[["X1","X2","M1"]].to_dict('records')
dictTestNA = dataTestNA[["X1","X2","M1"]].to_dict('records')

# Oracle XM
dataTest['PREDORACLEXM'] = [predRow(evDict, bnMissStruct, oracle = True, missingIndicator = True) for evDict in dictTest]

# Oracle X
dataTest['PREDORACLEX'] = [predRow(evDict, bnMissStruct, oracle = True, missingIndicator = False) for evDict in dictTest]

# Predict while having the structure
dataTest['PREDBNEM'] = [predRow(evDict, bnMissStruct, oracle = False, missingIndicator = True) for evDict in dictTestNA]

# Fit the model without the missing structure
bn = gum.fastBN("X1{0|1}->X2{0|1}->Y{0|1};X1->Y")
learner = gum.BNLearner(dataTrainNA,bn)
learner.useEM(1e-3)
learner.useSmoothingPrior()
learner.learnParameters(bn.dag())
dataTest['PREDBNE'] = [predRow(evDict, bn, oracle = False, missingIndicator = False) for evDict in dictTestNA]

# ------- Pattern submodels -------
## Fit PS0
dataTrainPS0 = dataTrainNA[dataTrainNA["M1"] == 0]
bnPS0 = gum.fastBN("X1{0|1}->X2{0|1}->Y{0|1};X1->Y")
learner = gum.BNLearner(dataTrainPS0,bnPS0)
learner.useSmoothingPrior()
learner.learnParameters(bnPS0.dag())

dataTrainPS1 = dataTrainNA[dataTrainNA["M1"] == 1]
bnPS1 = gum.fastBN("X2{0|1}->Y{0|1}")
learner = gum.BNLearner(dataTrainPS1,bnPS1)
learner.useSmoothingPrior()
learner.learnParameters(bnPS1.dag())

dataTest['PREDPS'] = 9.9
dictTestPS0 = dataTest.loc[dataTest["M1"] == 0, ["X1","X2","M1"]].to_dict('records')
dictTestPS1 = dataTest.loc[dataTest["M1"] == 1, ["X1","X2","M1"]].to_dict('records')

print("Predict PS 0")
predsPS0 =  [predRow(evDict, bnPS0, oracle = False, missingIndicator = False) for evDict in dictTestPS0]
print()
print("Predict PS 1")
predsPS1 =  [predRow(evDict, bnPS1, oracle = False, missingIndicator = False) for evDict in dictTestPS1]

dataTest.loc[dataTest["M1"] == 0, 'PREDPS'] = predsPS0
dataTest.loc[dataTest["M1"] == 1, 'PREDPS'] = predsPS1

print(dataTest.head)
dataTest.to_csv('predictedData.csv', index=False)

# TODO(): check why the predicted probabilities are so far from theoretical probabilities