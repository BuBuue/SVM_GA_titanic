#!/usr/bin/env python
# coding: utf-8

# In[44]:


#Import Lib
import math
import warnings
import pandas as pd
import numpy as np
import scipy as sp
import random as rd
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn import svm
from tqdm import tqdm
from mpl_toolkits import mplot3d
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV, RFE
from sklearn import metrics
from sklearn.model_selection import KFold,StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
pd.set_option('max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
#read file
train_df = pd.read_csv('Titanic/train.csv')
test_df = pd.read_csv('Titanic/test.csv')
train_df = train_df.drop(['Ticket', 'Cabin','PassengerId'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin','PassengerId'], axis=1)
combine = pd.concat([train_df, test_df])
#map Sex
combine['Sex'] = combine['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
#Map Embarked
combine.Embarked = combine.Embarked.fillna('S')
combine['Embarked'] = combine['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
#Fill na Age col
age_avg = combine['Age'].mean()
age_std = combine['Age'].std()
age_null_count = combine['Age'].isnull().sum()
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
combine['Age'][np.isnan(combine['Age'])] = age_null_random_list
combine['Age'] = combine['Age'].astype(int)
#Add title instead of using Name col
combine['Title'] = combine.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
combine['Title'] = combine['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
combine['Title'] = combine['Title'].replace('Mlle', 'Miss')
combine['Title'] = combine['Title'].replace('Ms', 'Miss')
combine['Title'] = combine['Title'].replace('Mme', 'Mrs')
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
combine['Title'] = combine['Title'].map(title_mapping)
combine['Title'] = combine['Title'].fillna(0)
combine = combine.drop(['Name'],axis=1)
#Add Fam size Col
combine['FamilySize'] = combine['SibSp'] + combine['Parch'] + 1
#Add Is Alone Col
combine['IsAlone'] = 0
combine.loc[combine['FamilySize'] == 1, 'IsAlone'] = 1
#Fill na Fare Col
combine['Fare'] = combine['Fare'].fillna(combine['Fare'].median()) 
###
combine['Survived'] = combine['Survived'].fillna(2)

##############
#    Model   #
#            #
##############

train = combine[(combine['Survived']== 1 )| (combine['Survived']== 0 )]
test = combine[(combine['Survived']== 2 )]
X_ = train.iloc[:, 1:]
Y = train[['Survived']]
X_test = test.iloc[:, 1:]


# In[46]:


#Part SVM
svc = SVC()
svc.fit(X_, Y)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_,Y) * 100, 2)
print('Accuracy from using only SVM : ',acc_svc)


# In[42]:


#PART SVM with GA
scaler = MinMaxScaler()
X = scaler.fit_transform(X_)

Cnt1 = len(X)

XY_0 = np.random.randint(0,2,X_.shape[1])

p_c = 0.9 # prob. of crossover
p_m = 0.1 # prob. of mutation
pop = 100 # initial population
gen =  50 # number of new Gen
kfold = 5

n_list = np.empty((0,len(XY_0)))

for i in range(pop):
    rd.shuffle(XY_0)
    n_list = np.vstack((n_list,XY_0))


Final_Best_in_Gen_X = []
Worst_Best_in_Gen_X = []

One_Final_Guy = np.empty((0,len(XY_0)+2)) 
One_Final_Guy_Final = []

Min_for_all_Gen_for_Mut_1 = np.empty((0,len(XY_0)+1))
Min_for_all_Gen_for_Mut_2 = np.empty((0,len(XY_0)+1))

Min_for_all_Gen_for_Mut_1_1 = np.empty((0,len(XY_0)+2))
Min_for_all_Gen_for_Mut_2_2 = np.empty((0,len(XY_0)+2))

Min_for_all_Gen_for_Mut_1_1_1 = np.empty((0,len(XY_0)+2))
Min_for_all_Gen_for_Mut_2_2_2 = np.empty((0,len(XY_0)+2))


Generation = 1
for i in tqdm(range(gen)):
    
    
    New_Pop = np.empty((0,len(XY_0))) #Generate new Parent
    
    All_in_Gen_X_1 = np.empty((0,len(XY_0)+1))
    All_in_Gen_X_2 = np.empty((0,len(XY_0)+1))
    
    Min_in_Gen_X_1= []
    Min_in_Gen_X_2 = []
    
    Save_Best_in_Gen_X = np.empty((0,len(XY_0)+1))
    Final_Best_in_Gen_X = []
    Worst_Best_in_Gen_X = []
    
    print('--------------------------------------')
    print("GEN: #:",Generation)
        
    Fam = 1
    
    for j in range(int(pop/2)):
        
        print('--------------------------------------')
        print("Fam: #:",Fam)
        
        # Tournament Selection
        
        Parents = np.empty((0,len(XY_0)))
        
        for i in range(2):
            
            Battle_Troops = []
            
            Candidate_1_idx = np.random.randint(0,len(n_list))
            Candidate_2_idx = np.random.randint(0,len(n_list))
            Candidate_3_idx = np.random.randint(0,len(n_list))
            
            while Candidate_1_idx==Candidate_2_idx:
                Candidate_1_idx = np.random.randint(0,len(n_list))
            while Candidate_2_idx==Candidate_3_idx:
                Candidate_3_idx = np.random.randint(0,len(n_list))
            while Candidate_1_idx==Candidate_3_idx:
                Candidate_3_idx = np.random.randint(0,len(n_list))



            Candidate_1 = n_list[Candidate_1_idx]
            Candidate_2 = n_list[Candidate_2_idx]
            Candidate_3 = n_list[Candidate_3_idx]
            
            Battle_Troops = [Candidate_1,Candidate_2,Candidate_3]
            
            # Candidate 1
             
            t = 0

            emp_list = []
            
            for i in Candidate_1:
                if Candidate_1[t] == 1:
                    emp_list.append(t)
                t = t+1
            
            NewX = X[:,emp_list]
            
            P_1 = 0
            
            kf = KFold(n_splits=kfold)
            
            for train_idx,test_idx in kf.split(X):
                
                X_train,X_test = NewX[train_idx],NewX[test_idx]
                Y_train,Y_test = Y.iloc[train_idx],Y.iloc[test_idx]
              
                model1 = SVC()
                model1.fit(X_train,Y_train)
                PL1 = model1.predict(X_test)
                
                AC1 = model1.score(X_test,Y_test)
                
                OF_So_Far_1 = 1-(AC1)
                
                P_1 += OF_So_Far_1
                
            OF_So_Far_W1 = P_1/kfold
            
            # Candidate 2
            
            t = 0

            emp_list = []
            
            for i in Candidate_2:
                if Candidate_2[t] == 1:
                    emp_list.append(t)
                t = t+1
            
            NewX = X[:,emp_list]
            
            P_2 = 0
            
            kf = KFold(n_splits=kfold)
            
            for train_idx,test_idx in kf.split(X):
                
                X_train,X_test = NewX[train_idx],NewX[test_idx]
                Y_train,Y_test = Y.iloc[train_idx],Y.iloc[test_idx]
                
                model1 = SVC()
                model1.fit(X_train,Y_train)
                PL1 = model1.predict(X_test)
                
                AC1 = model1.score(X_test,Y_test)
                
                OF_So_Far_2 = 1-(AC1)
                
                P_2 += OF_So_Far_2
                
            OF_So_Far_W2 = P_2/kfold
            
            # Candidate 3
            
            t = 0

            emp_list = []
            
            for i in Candidate_3:
                if Candidate_3[t] == 1:
                    emp_list.append(t)
                t = t+1
            
            NewX = X[:,emp_list]
            
            P_3 = 0
            
            kf = KFold(n_splits=kfold)
            
            for train_idx,test_idx in kf.split(X):
                X_train,X_test = NewX[train_idx],NewX[test_idx]
                Y_train,Y_test = Y.iloc[train_idx],Y.iloc[test_idx]
                
                model1 = SVC()
                model1.fit(X_train,Y_train)
                PL1 = model1.predict(X_test)
                
                AC1 = model1.score(X_test,Y_test)
                
                OF_So_Far_3 = 1-(AC1)
                
                P_3 += OF_So_Far_3
                
            OF_So_Far_W3 = P_3/kfold

            
            Champion_Candidate_1 = OF_So_Far_W1
            Champion_Candidate_2 = OF_So_Far_W2
            Champion_Candidate_3 = OF_So_Far_W3
            
            
            if Champion_Candidate_1 == min(Champion_Candidate_1,Champion_Candidate_2,Champion_Candidate_3):
                Winner = Candidate_1
                Champion = Champion_Candidate_1

            if Champion_Candidate_2 == min(Champion_Candidate_1,Champion_Candidate_2,Champion_Candidate_3):
                Winner = Candidate_2
                Champion = Champion_Candidate_2
            
            if Champion_Candidate_3 == min(Champion_Candidate_1,Champion_Candidate_2,Champion_Candidate_3):
                Winner = Candidate_3
                Champion = Champion_Candidate_3


            Parents = np.vstack((Parents,Winner))
 
        Parent_1 = Parents[0]
        Parent_2 = Parents[1]
        
        
        # Crossover
        
        Child_1 = np.empty((0,len(XY_0)))
        Child_2 = np.empty((0,len(XY_0)))
        
        
        
        
        Ran_CO_1 = np.random.rand()
        
        if Ran_CO_1 < p_c:
            
            
            Cr_1 = np.random.randint(0,len(XY_0))
            Cr_2 = np.random.randint(0,len(XY_0))
            
            while Cr_1 == Cr_2:
                Cr_2 = np.random.randint(0,len(XY_0))
                
            if Cr_1 < Cr_2:
                
                Med_Seg_1 = Parent_1[Cr_1:Cr_2+1]
                Med_Seg_2 = Parent_2[Cr_1:Cr_2+1]
                
                First_Seg_1 = Parent_1[:Cr_1]
                Sec_Seg_1 = Parent_1[Cr_2+1:]
                
                First_Seg_2 = Parent_2[:Cr_1]
                Sec_Seg_2 = Parent_2[Cr_2+1:]
                
                Child_1 = np.concatenate((First_Seg_1,Med_Seg_2,Sec_Seg_1))
                Child_2 = np.concatenate((First_Seg_2,Med_Seg_1,Sec_Seg_2))
                
            else:
                
                Med_Seg_1 = Parent_1[Cr_2:Cr_1+1]
                Med_Seg_2 = Parent_2[Cr_2:Cr_1+1]
                
                First_Seg_1 = Parent_1[:Cr_2]
                Sec_Seg_1 = Parent_1[Cr_1+1:]
                
                First_Seg_2 = Parent_2[:Cr_2]
                Sec_Seg_2 = Parent_2[Cr_1+1:]
                
                Child_1 = np.concatenate((First_Seg_1,Med_Seg_2,Sec_Seg_1))
                Child_2 = np.concatenate((First_Seg_2,Med_Seg_1,Sec_Seg_2))
        
        else:
            Child_1 = Parent_1
            Child_2 = Parent_2
            
        #Mutation Part
        Mutated_Child_1 = []
        
        t = 0
        
        for i in Child_1:
            
            Ran_Mut_1 = np.random.rand()
            
            if Ran_Mut_1 < p_m:
                
                if Child_1[t] == 0:
                    Child_1[t] = 1
                else:
                    Child_1[t] = 0
                t = t+1
                
                Mutated_Child_1 = Child_1
                
            else:
                Mutated_Child_1 = Child_1
                
        
        Mutated_Child_2 = []
        
        t = 0
        
        for i in Child_2:
            
            Ran_Mut_2 = np.random.rand()
            
            if Ran_Mut_2 < p_m:
                
                if Child_2[t] == 0:
                    Child_2[t] = 1
                else:
                    Child_2[t] = 0
                t = t+1
                
                Mutated_Child_2 = Child_2
                
            else:
                Mutated_Child_2 = Child_2
        
        #For mutated child_1
        
        t = 0

        emp_list = []
        
        for i in Mutated_Child_1:
            if Mutated_Child_1[t] == 1:
                emp_list.append(t)
            t = t+1
        
        NewX = X[:,emp_list]
        
        P_1 = 0
        
        kf = KFold(n_splits=kfold)
        
        for train_idx,test_idx in kf.split(X):
            X_train,X_test = NewX[train_idx],NewX[test_idx]
            Y_train,Y_test = Y.iloc[train_idx],Y.iloc[test_idx]
            
            model1 = SVC()
            model1.fit(X_train,Y_train)
            PL1 = model1.predict(X_test)
            
            AC1 = model1.score(X_test,Y_test)
            
            OF_So_Far_1 = 1-(AC1)
            
            P_1 += OF_So_Far_1
            
        OF_So_Far_M1 = P_1/kfold


        #For mutated child_2
        
        t = 0

        emp_list = []
        
        for i in Mutated_Child_2:
            if Mutated_Child_2[t] == 1:
                emp_list.append(t)
            t = t+1
        
        NewX = X[:,emp_list]
        
        P_2 = 0
        
        kf = KFold(n_splits=kfold)
        
        for train_idx,test_idx in kf.split(X):
            X_train,X_test = NewX[train_idx],NewX[test_idx]
            Y_train,Y_test = Y.iloc[train_idx],Y.iloc[test_idx]
            
            model1 = SVC()
            model1.fit(X_train,Y_train)
            PL1 = model1.predict(X_test)
            
            AC1 = model1.score(X_test,Y_test)
            
            OF_So_Far_2 = 1-(AC1)
            
            P_2 += OF_So_Far_2
            
        OF_So_Far_M2 = P_2/kfold
        
        
        print('--------------------------------------')
        print("FV for Mutated Child #1 at Gen",Generation,":",OF_So_Far_M1)
        print("FV for Mutated Child #2 at Gen",Generation,":",OF_So_Far_M2)


        All_in_Gen_X_1_1_temp = Mutated_Child_1[np.newaxis]
        
        All_in_Gen_X_1_1 = np.column_stack((OF_So_Far_M1,All_in_Gen_X_1_1_temp))
        
        All_in_Gen_X_2_2_temp = Mutated_Child_2[np.newaxis]
        
        All_in_Gen_X_2_2 = np.column_stack((OF_So_Far_M2,All_in_Gen_X_2_2_temp))
        
        
        All_in_Gen_X_1 = np.vstack((All_in_Gen_X_1,All_in_Gen_X_1_1))
        All_in_Gen_X_2 = np.vstack((All_in_Gen_X_2,All_in_Gen_X_2_2))
        
        
        Save_Best_in_Gen_X = np.vstack((All_in_Gen_X_1,All_in_Gen_X_2))
        New_Pop = np.vstack((New_Pop,Mutated_Child_1,Mutated_Child_2))
        R_1 = []
        t = 0
        for i in All_in_Gen_X_1:
            
            if(All_in_Gen_X_1[t,:1]) <= min(All_in_Gen_X_1[:,:1]):
                R_1 = All_in_Gen_X_1[t,:]
            t = t+1
        Min_in_Gen_X_1 = R_1[np.newaxis]
        R_2 = []
        t = 0
        for i in All_in_Gen_X_2:
            
            if(All_in_Gen_X_2[t,:1]) <= min(All_in_Gen_X_2[:,:1]):
                R_2 = All_in_Gen_X_2[t,:]
            t = t+1
        Min_in_Gen_X_2 = R_2[np.newaxis]
        Fam = Fam+1
    t = 0
    R_11 = []
    for i in Save_Best_in_Gen_X:
        
        if (Save_Best_in_Gen_X[t,:1]) <= min(Save_Best_in_Gen_X[:,:1]):
            R_11 = Save_Best_in_Gen_X[t,:]
        t = t+1
            
    Final_Best_in_Gen_X = R_11[np.newaxis]
    t = 0
    R_22 = []
    for i in Save_Best_in_Gen_X:
        
        if (Save_Best_in_Gen_X[t,:1]) >= max(Save_Best_in_Gen_X[:,:1]):
            R_22 = Save_Best_in_Gen_X[t,:1]
        t = t+1
            
    Worst_Best_in_Gen_X = R_22[np.newaxis]
    
    Darwin_Guy = Final_Best_in_Gen_X[:]
    Not_So_Darwin_Guy = Worst_Best_in_Gen_X[:]
    Darwin_Guy = Darwin_Guy[0:,1:].tolist()
    Not_So_Darwin_Guy = Not_So_Darwin_Guy[0:,1:].tolist()
    
    Best_1 = np.where((New_Pop==Darwin_Guy))
    Worst_1 = np.where((New_Pop==Not_So_Darwin_Guy))
    
    
    New_Pop[Worst_1] = Darwin_Guy
    
    n_list = New_Pop

    
    Min_for_all_Gen_for_Mut_1 = np.vstack((Min_for_all_Gen_for_Mut_1,
                                                   Min_in_Gen_X_1))
    Min_for_all_Gen_for_Mut_2 = np.vstack((Min_for_all_Gen_for_Mut_2,
                                                   Min_in_Gen_X_2))
    
    
    Min_for_all_Gen_for_Mut_1_1 = np.insert(Min_in_Gen_X_1,0,Generation)
    Min_for_all_Gen_for_Mut_2_2 = np.insert(Min_in_Gen_X_2,0,Generation)
    
    
    Min_for_all_Gen_for_Mut_1_1_1 = np.vstack((Min_for_all_Gen_for_Mut_1_1_1,
                                                       Min_for_all_Gen_for_Mut_1_1))
    
    Min_for_all_Gen_for_Mut_2_2_2 = np.vstack((Min_for_all_Gen_for_Mut_2_2_2,
                                                       Min_for_all_Gen_for_Mut_2_2))
    
    
    Generation = Generation+1
    
One_Final_Guy = np.vstack((Min_for_all_Gen_for_Mut_1_1_1,
                           Min_for_all_Gen_for_Mut_2_2_2))

t = 0
Final_Round = []

for i in One_Final_Guy:
    if(One_Final_Guy[t,1]) <= min(One_Final_Guy[:,1]):
        Final_Round = One_Final_Guy[t,:]
    t = t+1

One_Final_Guy_Final = Final_Round[np.newaxis]

print('--------------------------------------')
print("Min of all Gen:",One_Final_Guy_Final)

print('--------------------------------------')
print("Final Solution:",One_Final_Guy_Final[:,2:])
print("Highest Acc:",(1-One_Final_Guy_Final[:,1]))
A11 = One_Final_Guy_Final[:,2:][0]
t = 0
emp_list = []
for i in A11:
    if A11[t] == 1:
        emp_list.append(t)
    t = t+1
print('--------------------------------------')
print("Features that included are:",emp_list)

