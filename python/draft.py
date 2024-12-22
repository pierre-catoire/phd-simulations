import pandas as pd
import pyAgrum as gum

def predRow(rowDict, model, oracle=True, missingIndicator=True):

    if (oracle == False) & (rowDict["M1"] == 1):
        rowDict.pop("X1")

    if missingIndicator == False:
        rowDict.pop("M1")

    ie = gum.LazyPropagation(model)
    ie.setEvidence(rowDict)
    ie.makeInference()
    return ie.posterior("Y")[1]

bn = gum.fastBN("X1{0|1}->X2{0|1}->Y{0|1};X1->Y; X1 -> M1{0|1}")

theta = [0.5, 0.3, 0.4, 0.7, 0.5, 0.3, 0.8]
phi = [0.3, 0.2, 0.4]

bn.cpt("X1")[{}] = [theta[0], 1-theta[0]]
bn.cpt("X2")[{"X1": 0}] = [theta[1], 1-theta[1]]
bn.cpt("X2")[{"X1": 1}] = [theta[2], 1-theta[2]]
bn.cpt("Y")[{"X1": 0, "X2": 0}] = [theta[3], 1-theta[3]]
bn.cpt("Y")[{"X1": 0, "X2": 1}] = [theta[4], 1-theta[4]]
bn.cpt("Y")[{"X1": 1, "X2": 0}] = [theta[5], 1-theta[5]]
bn.cpt("Y")[{"X1": 1, "X2": 1}] = [theta[6], 1-theta[6]]

bn.cpt("M1")[{"X1": 0}] = [phi[1], 1 - phi[1]]
bn.cpt("M1")[{"X1": 1}] = [phi[2], 1 - phi[2]]


rowDict = {"X2": 1, "M1": 1}
ie = gum.LazyPropagation(bn)
ie.setEvidence(rowDict)
ie.makeInference()
print(ie.posterior("Y")[1])
print(predRow({"X1": 1,"X2": 1, "M1": 1},model = bn,oracle = False,missingIndicator = True))