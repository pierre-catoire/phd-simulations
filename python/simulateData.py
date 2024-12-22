# Objective:
# - simulate datasets from various structures, theta and phi parameters
#
# Output:
# - simulatedData.csv, with the values of variables, parameters, missprop, iteration

# preamble
import pyAgrum as gum
import pandas as pd
from time import process_time

# Functions

def predRow(rowDict, model, oracle=True, missingIndicator=True):
    # Predict a row (using evidence passed as a dictionary: rowDict).
    # - if Oracle is True, the X1 value will be kept event if M1 = 1
    # - if missingIndicator is True, M1 will be kept

    if (oracle == False) & (rowDict["M1"] == 1):
        rowDict.pop("X1")

    if missingIndicator == False:
        rowDict.pop("M1")

    ie = gum.LazyPropagation(model)
    ie.setEvidence(rowDict)
    ie.makeInference()
    return ie.posterior("Y")[1]


# Generate datasets
structureStrings = {"S3_1": "X1{0|1}->X2{0|1}->Y{0|1};X1->Y; M1{0|1}; X3{0|1}",
                    "S3_2": "X1{0|1}->X2{0|1}->Y{0|1};X1->Y; X2->M1{0|1}; X3{0|1}",
                    "S3_3": "X1{0|1}->X2{0|1}->Y{0|1};X1->Y; X3{0|1}->M1{0|1};X3->X2",
                    "S3_4": "X1{0|1}->X2{0|1}->Y{0|1};X1->Y; X1->M1{0|1}; X3{0|1}",
                    "S3_5": "X1{0|1}->X2{0|1}->Y{0|1};X1->Y; Y->M1{0|1}; X3{0|1}",
                    "S3_6": "X1{0|1}->X2{0|1}->Y{0|1};X1->Y; Y->M1{0|1}<-X1; X3{0|1}"}

missProp = 0.3
sSize = 1000
nIterations = 100

theta = [0.5,
         0.6,
         0.3,
         0.5,
         0.2,
         0.6,
         0.3,
         0.25,
         0.45,
         0.5]

phi = [missProp * 1,
       missProp * 0.8,
       missProp * 1.2,
       missProp * 0.4,
       missProp * 1.8,
       missProp * 0.6,
       missProp * 1.5]

bnList = {}

data = pd.DataFrame(columns=["X1","X2","X3","M1","Y",
                             "theta0","theta1","theta2","theta3","theta4","theta5","theta6","theta7","theta8","theta9",
                             "phi0","phi1","phi2","phi3","phi4","phi5","phi6","missprop"])

dataList = {}

start_time = process_time()
for structureLabel in structureStrings.keys():
    for iteration in range(nIterations):
        # Define the BN
        bn = gum.fastBN(structureStrings[structureLabel])

        # Add cpts from theta
        if structureLabel != "S3_3":
            bn.cpt("X1")[{}] = [theta[0], 1-theta[0]]
            bn.cpt("X2")[{"X1": 0}] = [theta[1], 1-theta[1]]
            bn.cpt("X2")[{"X1": 1}] = [theta[2], 1-theta[2]]
            bn.cpt("Y")[{"X1": 0, "X2": 0}] = [theta[3], 1-theta[3]]
            bn.cpt("Y")[{"X1": 0, "X2": 1}] = [theta[4], 1-theta[4]]
            bn.cpt("Y")[{"X1": 1, "X2": 0}] = [theta[5], 1-theta[5]]
            bn.cpt("Y")[{"X1": 1, "X2": 1}] = [theta[6], 1-theta[6]]
            bn.cpt("X3")[{}] = [theta[9], 1 - theta[9]]

        else:
            bn.cpt("X1")[{}] = [theta[0], 1 - theta[0]]
            bn.cpt("X3")[{}] = [theta[9], 1 - theta[9]]
            bn.cpt("X2")[{"X1": 0, "X3":0}] = [theta[1], 1 - theta[1]]
            bn.cpt("X2")[{"X1": 1, "X3":0}] = [theta[2], 1 - theta[2]]
            bn.cpt("X2")[{"X1": 0, "X3": 1}] = [theta[7], 1 - theta[7]]
            bn.cpt("X2")[{"X1": 1, "X3": 1}] = [theta[8], 1 - theta[8]]
            bn.cpt("Y")[{"X1": 0, "X2": 0}] = [theta[3], 1 - theta[3]]
            bn.cpt("Y")[{"X1": 0, "X2": 1}] = [theta[4], 1 - theta[4]]
            bn.cpt("Y")[{"X1": 1, "X2": 0}] = [theta[5], 1 - theta[5]]
            bn.cpt("Y")[{"X1": 1, "X2": 1}] = [theta[6], 1 - theta[6]]

        # add CPT of M from phi
        if structureLabel == "S3_1":
            bn.cpt("M1")[{}] = [phi[0], 1 - phi[0]]

        elif structureLabel == "S3_2":
            bn.cpt("M1")[{"X2":0}] = [phi[1], 1 - phi[1]]
            bn.cpt("M1")[{"X2":1}] = [phi[2], 1 - phi[2]]

        elif structureLabel == "S3_3":
            bn.cpt("M1")[{"X3": 0}] = [phi[1], 1 - phi[1]]
            bn.cpt("M1")[{"X3": 1}] = [phi[2], 1 - phi[2]]

        elif structureLabel == "S3_4":
            bn.cpt("M1")[{"X1": 0}] = [phi[1], 1 - phi[1]]
            bn.cpt("M1")[{"X1": 1}] = [phi[2], 1 - phi[2]]

        elif structureLabel == "S3_5":
            bn.cpt("M1")[{"Y": 0}] = [phi[1], 1 - phi[1]]
            bn.cpt("M1")[{"Y": 1}] = [phi[2], 1 - phi[2]]

        elif structureLabel == "S3_6":
            bn.cpt("M1")[{"X1": 0, "Y": 0}] = [phi[3], 1 - phi[3]]
            bn.cpt("M1")[{"X1": 0, "Y": 1}] = [phi[4], 1 - phi[4]]
            bn.cpt("M1")[{"X1": 1, "Y": 0}] = [phi[5], 1 - phi[5]]
            bn.cpt("M1")[{"X1": 1, "Y": 1}] = [phi[6], 1 - phi[6]]

        bnList[structureLabel] = bn

        g=gum.BNDatabaseGenerator(bn)
        g.drawSamples(sSize)
        df = g.to_pandas()
        df['missprop'] = missProp
        df['theta0'] = theta[0]
        df['theta1'] = theta[1]
        df['theta2'] = theta[2]
        df['theta3'] = theta[3]
        df['theta4'] = theta[4]
        df['theta5'] = theta[5]
        df['theta6'] = theta[6]
        df['theta7'] = theta[7]
        df['theta8'] = theta[8]
        df['theta9'] = theta[9]
        df['phi0'] = phi[0]
        df['phi1'] = phi[1]
        df['phi2'] = phi[2]
        df['phi3'] = phi[3]
        df['phi4'] = phi[4]
        df['phi5'] = phi[5]
        df['phi6'] = phi[6]
        df['structureLabel'] = structureLabel
        df['iteration'] = iteration

        # TODO(): get theoretical probabilities using predRow()
        # - P(Y|X,M): predRow(evDict, bn, Oracle = True, missingIndicator = True)
        # - P(Y|X): predRow(evDict, bn, Oracle = True, missingIndicator = False)
        # - P(Y|E,M): predRow(evDict, bn, Oracle = False, missingIndicator = True)
        # - P(Y|E): predRow(evDict, bn, Oracle = False, missingIndicator = False)



        dataList[structureLabel+"_"+str(iteration)] = df

data = pd.concat(dataList, ignore_index=True)

data.to_csv('simulatedData.csv', index=False)

end_time = process_time()
print("Elapsed time:", end_time-start_time)