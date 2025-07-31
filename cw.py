import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from collections import Counter
import time

# Load dataset
heading =["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "type"]
df = pd.read_csv('glass.data.csv', names = heading)
df.drop(["Id"], axis =1, inplace = True)

# Train and split the data
id = 39542335
features = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
X = df[["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]]
y = df['type']

X_train, X_test, y_train, y_test = train_test_split(X,y , random_state=id,test_size=0.20)

#converting the dat to numpy
X_train = X_train.to_numpy(dtype=float)
X_test = X_test.to_numpy(dtype=float)

# Accuracy function
def newer_acc(y_test, newY_pred):
    y_test = np.array(y_test)
    newY_pred = np.array(newY_pred)
    correct = 0
    for i in range(len(y_test)):
        if y_test[i] == newY_pred[i]:
            correct += 1
    return correct / float(len(y_test))

#KNN CLASSIFIER 
def euclean_dist(a,b):
    return np.sqrt(np.sum((np.array(a) - np.array(b))**2))

def newKNN(train_data, train_labels, testings, k):
    distance = []
    for i in range (len(train_data)):
        dis = euclean_dist(testings, train_data[i])
        distance.append((dis, train_labels.iloc[i]))
    
    distance.sort(key= lambda x: x[0])
    k_nearest = [label for _, label in distance[:k]]
    return Counter(k_nearest).most_common(1)[0][0]

#KNN Prediction
pred_start = time.time()
k = 1
newY_pred = [newKNN(X_train, y_train, sample,k) for sample in X_test]
pred_end = time.time()
total_pred = pred_end - pred_start

#KNN testing
test_start = time.time()
knn_acc = newer_acc(y_test, newY_pred)
test_end = time.time()
total_test = test_end - test_start
cm1 = confusion_matrix(y_test, newY_pred, labels=np.unique(y_test))

print("KNN RESULTS:")
print(f"Manual Acc = {knn_acc * 100:.2f}%")
print(f'Predicition time: {total_pred:.10f} secs')
print(f'Test time: {total_test:.10f} secs')
print("confusion matrix = ", cm1)


#DECISION TREE CLASSIFIER
#NODE CLASS
class Node(): #represents the node of the tree
    def __init__(self, index=None, threshold=None, left=None, right=None, info=None, value=None):
        
        self.index = index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info = info
        self.value = value

#DECISION TREE CLASS
class DecisionTree():
    def __init__(self, sample_split=2, max_depth=3):
        #root of tree
        self.root = None
        self.sample_split = sample_split
        self.max_depth = max_depth

    def tree(self, glass_set, curr_depth=0):

        X_train,y_train = glass_set[:, :-1], glass_set[:, -1]
        sampleNum, featureNum = np.shape(X_train)

        if sampleNum >= self.sample_split and curr_depth <= self.max_depth:
            best_split = self.get_best_split(glass_set, sampleNum, featureNum)

            if best_split['info'] > 0:
                left = self.tree(best_split['glass_set_left'], curr_depth +1)
                right = self.tree(best_split['glass_set_right'], curr_depth +1)

                return Node(best_split['index'], best_split['threshold'], left, right, best_split['info'])
            
        leaf = self.calc_leaf_val(y_train)
        return Node(value=leaf)
    
    def get_best_split(self, glass_set, sampleNum, featureNum):
        best_split = {}
        max_info = -float('inf')

        for index in range(featureNum):
            featureVal = glass_set[:, index]
            possible_thresholds = np.unique(featureVal)

            for threshold in possible_thresholds:
                glass_set_left, glass_set_right = self.split(glass_set, index, threshold)

                if len(glass_set_left) > 0 and len(glass_set_right) > 0:
                    y, left_y, right_y = glass_set[:, -1], glass_set_left[:,-1], glass_set_right[:,-1]

                    current_info = self.information(y, left_y, right_y, 'entropy')

                    if current_info > max_info:
                        max_info = current_info
                        best_split['index'] = index
                        best_split['threshold'] = threshold
                        best_split['glass_set_left'] = glass_set_left
                        best_split['glass_set_right'] = glass_set_right
                        best_split['info'] = current_info
        
        return best_split
    
    def split(self, glass_set, index, threshold):
        # return glass_set_left, glass_set_right
        left_split = glass_set[:, index] <= threshold
        right_split = glass_set[:, index] > threshold
        return glass_set[left_split], glass_set[right_split]
    
    def information(self, parent, left_child, right_child, mode='entropy'):
        if len(left_child) == 0 or len(right_child) == 0:
            return 0
        
        left_weight = len(left_child)/len(parent)
        right_weight = len(right_child)/len(parent)

        if mode == 'gini':
            gained = self.gini_index(parent) - (left_weight * self.gini_index(left_child) + right_weight * self.gini_index(right_child))
        else:
            gained = self.entropy(parent) - (left_weight * self.entropy(left_child) + right_weight * self.entropy(right_child))
        return gained
    
    def entropy(self, y):
        if len(y) == 0:
            return 0
        class_label = np.unique(y)
        entropy = 0
        for cls in class_label:
            parent_class = len(y[y == cls]) / len(y)
            if parent_class > 0:
                entropy += -(parent_class) * np.log2(parent_class)
        return entropy
    
    def gini_index(self, y):
        class_label = np.unique(y)
        gini = 0
        for cls in class_label:
            parent_class = len(y[y==cls]) / len(y)
            gini += parent_class **2
        return 1 - gini
    
    def calc_leaf_val(self, Y):
        return np.argmax(np.bincount(Y.astype(int)))
        
    def print_tree(self, tree=None, indent=' '):
        if not tree:
            tree = self.root
        
        if tree.value is not None:
            print(tree.value)

        else:
            print('X_' + str(tree.index), '<=', tree.threshold, '?', tree.info)
            print('%sleft:' % (indent), end='')
            self.print_tree(tree.left, indent + indent)
            print('%sright:' % (indent), end='')
            self.print_tree(tree.right, indent + indent)

    def Dec_fit(self,X_train,y_train):
        glass_set = np.concatenate((X_train,y_train.to_numpy().reshape(-1,1)), axis=1)
        self.root = self.tree(glass_set)

    def Dec_predict(self,X_train):
        prediction = [self.make_preds(x, self.root) for x in X_train]
        return prediction
    
    def make_preds(self, x, tree):
        if tree.value != None: 
            return tree.value
        
        featureVal = x[tree.index]
        if featureVal <= tree.threshold:
            return self.make_preds(x, tree.left)
        else:
            return self.make_preds(x, tree.right)

#Decision tree training
training_start = time.time()
clf = DecisionTree(sample_split=5, max_depth=5) # changed the depth and split
clf.Dec_fit(X_train, y_train)
training_end = time.time()
total_training = training_end - training_start

#Decision tree predition
predict_start = time.time()
newY = clf.Dec_predict(X_test)
predict_end = time.time()
total_predict = predict_end - predict_start

#Decision tree testing
testing_start = time.time()
accu = newer_acc(y_test, newY)
testing_end = time.time()
total_testing = testing_end - testing_start
cm2 = confusion_matrix(y_test, newY, labels=np.unique(y_test))

print("DECISION TREE RESULTS:")
print(f"Acc = {accu * 100:.2f}%")
print(f'Training time: {total_training:.10f} secs')
print(f'Predicition time: {total_predict:.10f} secs')
print(f'Test time: {total_testing:.10f} secs')
print("confusion matrix = ", cm2)
        
## NAIVE BAYES CLASSIFIER
unique_class, class_counts = np.unique(y_train, return_counts=True)
prior_y = class_counts / len(y_train)

featuresNum = X_train.shape[1]
classNum = len(unique_class)

new_mean = np.zeros((classNum, featuresNum))
new_variance = np.zeros((classNum, featuresNum))

means = np.asarray(new_mean)
vard = np.asarray(new_variance)

for i, label in enumerate(unique_class):
    data = X_train[y_train == label]
    means[i] = data.mean(axis=0)
    vard[i] = data.var(axis=0)

def NB_probability(x, mu, sigma2, epsilon=1e-6):
    sigma2 = np.maximum(sigma2, epsilon)
    a = (1/np.sqrt(2 * (np.pi) * sigma2))
    b = np.exp(-0.5 * ((x - mu) **2) / sigma2)
    return a * b

def new_NB_pred(test):
    pY = np.zeros(classNum)
    for i in range(classNum):
        pX = np.prod(NB_probability(test, means[i], vard[i]))
        pY[i] = pX * prior_y[i]
    return unique_class[np.argmax(pY)]

def NB_fitt(X_train, y_train):
    return means, vard, prior_y

# NB Training
train_start = time.time()
NB_fitt(X_train, y_train)
train_end = time.time()
train_total = train_end - train_start

#NB Predicition
start_pred = time.time()
g_pred = [int(new_NB_pred(row)) for row in X_test]
end_pred = time.time()
pred_total = end_pred - start_pred

#NB Testing
start_test = time.time()
ac = newer_acc(g_pred,  y_test)
end_test = time.time()
test_total = end_test - start_test
types = np.unique(y_test)
cm = confusion_matrix(y_test, g_pred)

print("NAIVE BAYES RESULTS:")
print(f"accuracy val = {ac*100:.2f}%")
print(f'Training time: {train_total:.10f} secs')
print(f'Predicition time: {pred_total:.10f} secs')
print(f'Test time: {test_total:.10f} secs')
print("confusion matrix = ", cm)

## SVM CLASSIFIER
class SVM:
    def __init__(self, epoches=1000, learning_rate=0.001, alpha = 0.01):
        self.epoches = epoches
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.w = None
        self.b = None

    def svm_fit(self, X, y):
        self.ones = np.ones(X.shape[0])
        self.b = 0
        samples = X.shape[0]
        featuresNum = X.shape[1]
        self.w = np.zeros((1, featuresNum))
        y_ = np.where(y == self.b,-1,1)

        # Train a binary SVM for each class
        for _ in range(self.epoches):
            for i, x_i in enumerate(X):
                loss = y_[i] * (np.dot(x_i, self.w.T) - self.b) >= 1
            if loss:
                self.w -= self.learning_rate * (2* self.alpha * self.w)
            else:
                self.w -= self.learning_rate * (2 * self.alpha * self.w - np.dot(x_i, y_[i]))
                self.b -= self.learning_rate * y_[i]
           
    def svm_predict(self, X):
        return np.sign(np.dot(X, self.w.T) - self.b)

#SVM Training
svm_train = time.time()
sclf = SVM()
sclf.svm_fit(X_train, y_train)
svm_train_end = time.time()
strain_total = svm_train_end - svm_train

#SVM Predicition
svm_pred = time.time()
s_pred = sclf.svm_predict(X_test)
svm_pred_end = time.time()
spred_total = svm_pred_end - svm_pred

#SVM Testing
svm_test = time.time()
s_acc = newer_acc(s_pred, y_test)
stest_end = time.time()
stest_total = stest_end - svm_test
cm3 = confusion_matrix(y_test, s_pred)

print("SVM RESULTS:")
print(f"accuracy val = {s_acc*100:.2f}%")
print(f'Training time: {strain_total:.10f} secs')
print(f'Prediction time: {spred_total:.10f} secs')
print(f'Test time: {stest_total:.10f} secs')
print("Confusion matrix = ", cm3)
