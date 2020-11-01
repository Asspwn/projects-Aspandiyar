import pandas as pd
import numpy as np
from sys import *

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score, confusion_matrix, fbeta_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import warnings
warnings.filterwarnings("ignore")

# uses the sklearn confusion matrix and creates a nice, readable string
def nice_cf_str(cf):
	cf_str = 'a\tb   <---- classified as\n'
	cf_str += '%d\t%d\ta = irrelevant\n'%(cf[0,0], cf[0,1])
	cf_str += '%d\t%d\tb = relevant\n'%(cf[1,0], cf[1,1])
	return cf_str

data_dir = argv[1]
given_threshold = float(argv[2])
beta = float(argv[3])

# setting parameters, also for the numer of folds
random_state = 0
folds = 10

# loading the training set based on the path given with the command line arguments
training_set = pd.read_csv('%straining.csv'%(data_dir), sep=' ')

# collecting all column names that are not the class
# consequently those are the feature columns
features = []
for col in training_set.columns:
	if col != 'class':
		features.append(col)

# create the feature set and class vector
X = np.array(training_set[features])
y = np.array(training_set['class'])

# initiating the cross-validation and classification model
tenFoldCV = StratifiedKFold(n_splits=folds, random_state=random_state, shuffle=True)
clf = GradientBoostingClassifier(random_state=0, n_estimators=200, learning_rate=0.18)
#clf = RandomForestClassifier(random_state=0, n_estimators=20)

# lists that collect the area under curve values
# for ROC-curve and Precision-Recall curve
auROCs = []
auPRCs = []

# lists that collect the best decision thresholds
# based on the default F1 measure (beta=1)
dec_thresholds_f1 = []
# based on the F-measure using the beta provided by command line argument
dec_thresholds_fb = []

for fold, (train, test) in enumerate(tenFoldCV.split(X, y)):
	print('fold %d ...'%(fold+1))
	
	# training the classification model
	model = clf.fit(X[train], y[train])
	# using the model to calculate classification probabilities
	probas = model.predict_proba(X[test])
	
	# area under Precision-Recall-curve
	precision, recall, p_r_thresholds = precision_recall_curve(y[test], probas[:,1])
	prc_auc = auc(recall, precision)
	auPRCs.append(prc_auc)
	
	# area under ROC-curve
	fpr, tpr, thresholds = roc_curve(y[test], probas[:, 1])
	roc_auc = auc(fpr, tpr)
	auROCs.append(roc_auc)
	
	# for 100 different thresholds the F1 and F_beta measures are computed
	threshold_f1 = []
	threshold_fb = []
	for threshold in [float(i)/100.0 for i in range(101)]:
		# based on the given threshold create the vector of predicted values
		temp_pred = [1 if p > threshold else 0 for p in probas[:, 1]]
		# calculate F1 measure
		f1 = f1_score(y[test], temp_pred)
		# calculate F_beta measure
		fb = fbeta_score(y[test], temp_pred, beta=beta)
		# collect all thresholds and the F-measures
		threshold_f1.append((threshold, f1))
		threshold_fb.append((threshold, fb))
	# sort thresholds based on the F-measure and select the threshold
	# leading to the best F-measure
	threshold_f1 = sorted(threshold_f1, key=lambda x: x[1], reverse=True)
	dec_thresholds_f1.append(threshold_f1[0][0])
	threshold_fb = sorted(threshold_fb, key=lambda x: x[1], reverse=True)
	dec_thresholds_fb.append(threshold_fb[0][0])

print('Performance on Training Set:')
print('auROC: %.3f'%(np.mean(auROCs)))
print('auPRC: %.3f\n'%(np.mean(auPRCs)))

# train one final model on the whole training set
final_model = clf.fit(X, y)

# load the unseen testing set and calculate the class probabilities on this unseen data
testing_set = pd.read_csv('%stesting.csv'%(data_dir), sep=' ')
final_test_X = testing_set[features]
final_test_y = testing_set['class']
final_probas = final_model.predict_proba(final_test_X)

# calculating area under the curves (ROC, Precision-Recall)
fpr, tpr, thresholds = roc_curve(final_test_y, final_probas[:, 1])
roc_auc = auc(fpr, tpr)
precision, recall, thresholds = precision_recall_curve(final_test_y, final_probas[:,1])
prc_auc = auc(recall, precision)

print('Performance on Test Set:')
print('auROC: %.3f'%(roc_auc))
print('auPRC: %.3f\n'%(prc_auc))

# decision threshold are fitted by two different betas
# the default beta=1 and a second beta defined by the user
print('Fitted Decision Threshold based on F1: %.3f'%(np.mean(dec_thresholds_f1)))
print('Fitted Decision Threshold based on F (beta=%s): %.3f'%(str(beta), np.mean(dec_thresholds_fb)))

# printing confusion matrices based on different decision thresholds:
# default: 0.5
# fitted decision threshold based on F1
# fitted decision threshold based on F_beta


def_pred = [1 if p > 0.5 else 0 for p in final_probas[:, 1]]
print('\n\nConfusion Matrix on default decision threshold 0.5\n')
print(nice_cf_str(confusion_matrix(final_test_y, def_pred)))

# the fitted decision threshold
fdt = np.mean(dec_thresholds_f1)
fitted_pred = [1 if p > fdt else 0 for p in final_probas[:, 1]]
print('\n\nConfusion Matrix on fitted decision threshold using F1: %.3f\n'%(fdt))
print(nice_cf_str(confusion_matrix(final_test_y, fitted_pred)))

# the fitted decision threshold based on the given beta
fbetath = np.mean(dec_thresholds_fb)
new_pred = [1 if p > fbetath else 0 for p in final_probas[:, 1]]
print('\n\nConfusion Matrix on fitted decision threshold using F%s: %.3f\n'%(str(beta),fbetath))
print(nice_cf_str(confusion_matrix(final_test_y, new_pred)))

# the given decision threshold
new_pred = [1 if p > given_threshold else 0 for p in final_probas[:, 1]]
print('\n\nConfusion Matrix on given decision threshold %.3f\n'%(given_threshold))
print(nice_cf_str(confusion_matrix(final_test_y, new_pred)))



















