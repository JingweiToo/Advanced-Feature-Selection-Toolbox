# Jx-AFST : Advanced Feature Selection Toolbox

---
> "Toward Talent Scientist: Sharing and Learning Together"
>  --- [Jingwei Too](https://jingweitoo.wordpress.com/)
---

![Wheel](https://github.com/JingweiToo/Advanced-Feature-Selection-Toolbox/blob/main/Capture.JPG)


## Introduction

* This toolbox offers several advanced wrapper feature selection methods
* Source code of these methods are written based on pseudocode & paper


## Usage
The main function *jfs* is adopted to perform feature selection. You may switch the algorithm by changing the 'issa' to [other abbreviations](/README.md#list-of-available-advanced-feature-selection-methods)
* If you wish to use improved salp swarm algorithm ( ISSA ) then you may write
```code
from AFS.issa import jfs
```
* If you want to use time varying binary salp swarm algorithm ( TVBSSA ) then you may write
```code
from AFS.tvbssa import jfs
```


## Input
* *feat*   : feature vector matrix ( Instance *x* Features )
* *label*  : label matrix ( Instance *x* 1 )
* *opts*   : parameter settings
    + *N* : number of solutions / population size ( *for all methods* )
    + *T* : maximum number of iterations ( *for all methods* )
    + *k* : *k*-value in *k*-nearest neighbor 


## Output
* *Acc*  : accuracy of validation model
* *fmdl* : feature selection model ( It contains several results )
    + *sf* : index of selected features
    + *nf* : number of selected features
    + *c*  : convergence curve
    
    
### Example : Improved Salp Swarm Algorithm ( ISSA ) 
```code 
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from AFS.issa import jfs   # change this to switch algorithm 
import matplotlib.pyplot as plt


# load data
data  = pd.read_csv('ionosphere.csv')
data  = data.values
feat  = np.asarray(data[:, 0:-1])   # feature vector
label = np.asarray(data[:, -1])     # label vector

# split data into train & validation (70 -- 30)
xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.3, stratify=label)
fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}

# parameter
k    = 5     # k-value in KNN
N    = 10    # number of salps
T    = 100   # maximum number of iterations
opts = {'k':k, 'fold':fold, 'N':N, 'T':T}

# perform feature selection
fmdl = jfs(feat, label, opts)
sf   = fmdl['sf']

# model with selected features
num_train = np.size(xtrain, 0)
num_valid = np.size(xtest, 0)
x_train   = xtrain[:, sf]
y_train   = ytrain.reshape(num_train)  # Solve bug
x_valid   = xtest[:, sf]
y_valid   = ytest.reshape(num_valid)  # Solve bug

mdl       = KNeighborsClassifier(n_neighbors = k) 
mdl.fit(x_train, y_train)

# accuracy
y_pred    = mdl.predict(x_valid)
Acc       = np.sum(y_valid == y_pred)  / num_valid
print("Accuracy:", 100 * Acc)

# number of selected features
num_feat = fmdl['nf']
print("Feature Size:", num_feat)

# plot convergence
curve   = fmdl['c']
curve   = curve.reshape(np.size(curve,1))
x       = np.arange(0, opts['T'], 1.0) + 1.0

fig, ax = plt.subplots()
ax.plot(x, curve, 'o-')
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Fitness')
ax.set_title('ISSA')
ax.grid()
plt.show()
```


## Requirement

* Python 3 
* Numpy
* Pandas
* Scikit-learn
* Matplotlib


## List of available advanced feature selection methods
* The extra parameters represent the parameter(s) other than population size and maximum number of iterations
* Click on the name of method to view the extra parameter(s)
* Use the *opts* to set the specific parameter(s)


| No. | Abbreviation | Name                                                                                        | Year | Extra Parameters |
|-----|--------------|---------------------------------------------------------------------------------------------|------|------------------|
| 02  | tvbssa       | Time Varying Binary Salp Swarm Algorithm                                                    | 2020 | No               |
| 01  | issa         | [Improved Salp Swarm Algorithm](/Description.md#improved-salp-swarm-algorithm-issa)         | 2020 | Yes              |

