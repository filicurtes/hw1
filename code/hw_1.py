import pandas as pd 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
import numpy as np 
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def sklearn_bunch_to_df(dataset_bunch):
    df=pd.DataFrame(dataset_bunch.data,columns=dataset_bunch.feature_names)
    df['class']=pd.Series(dataset_bunch.target)
    return df
df=sklearn_bunch_to_df(load_wine())


##get classes
y=df['class'].values
df=df.drop(columns='class')
##get values from df
X=df.values
print(df.columns)

def features_selection(dataframe,ncol1,ncol2):
    index1=ncol1-1
    index2=ncol2
    dataframe=dataframe[:,[index1,index2]]
    return dataframe

##column index selection
index1=1
index2=2
X=features_selection(X,index1,index2)

##plot features space
fig,ax=plt.subplots()
ax.scatter(X[:,0],X[:,1],c=y,cmap='spring')
ax.set_title('Classes distribution in R^2 with these 2 features')
ax.set_xlabel(df.columns[index1-1])
ax.set_ylabel(df.columns[index2-1])
#plt.show()

##splitting Data in sets
X_train,X_valtest,y_train,y_valtest=train_test_split(X,y,train_size=0.3,stratify=y)
X_val,X_test,y_val,y_valtest=train_test_split(X_valtest,y_valtest,train_size=0.29,stratify=y_valtest)

def k_nn(X_train,y_train,X_val,y_val) :
    ## K-nearest selection
    acc=[]
    k_values=[]
    for k in range(1,8,2):

        weights='boh'

        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train,y_train)
        y_pred=knn.predict(X_val)

        def plot_dec_bound(X,y,k,weight):
            h=0.02
            k_values.append(k)
            cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
            cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
            Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.figure()
            plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

            # Plot also the training points
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,edgecolor='k', s=20)
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.title("3-Class classification (k = %i, weights = '%s')"% (k, weight))

            #plt.show()

            acc.append(accuracy_score(y_val,y_pred))

        plot_dec_bound(X_val,y_val,k,weights)

    fig,ax=plt.subplots()
    ax.plot(k_values,acc,marker='o')
    ax.set_title('Accuracy trend by K-values')
    ax.set_xlabel('K-neighbours number')
    ax.set_ylabel('Accuracy')
    plt.show()
#k_nn(X_train,y_train,X_val,y_val)


def lin_svm(X_train,y_train,X_val,y_val):
    c=0.0001
    acc=[]
    c_values=[]
    for i in range(1,7):
        c=c*10
        c_values.append(c)
        svm=SVC(kernel='linear',C=c)
        svm.fit(X_train,y_train)
        y_pred=svm.predict(X_val)
        acc.append(accuracy_score(y_val,y_pred))
        
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        h=0.02
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X_val[:, 0].min() - 1, X_val[:, 0].max() + 1
        y_min, y_max = X_val[:, 1].min() - 1, X_val[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap=cmap_bold,edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification (k = %i, weights = '%s')"% (c, 'uniform'))

        
    fig,ax=plt.subplots()
    ax.plot(c_values,acc,marker='o')
    ax.set_title('Accuracy trend by C-values')
    ax.set_xlabel('C-neighbours number')
    ax.set_ylabel('Accuracy')
    ax.set_xscale('log')
    plt.show()

lin_svm(X_train,y_train,X_val,y_val)

def lin_svm(X_train,y_train,X_val,y_val):
    c=0.0001
    acc=[]
    c_values=[]
    for i in range(1,7):
        c=c*10
        c_values.append(c)
        svm=SVC(C=c)
        svm.fit(X_train,y_train)
        y_pred=svm.predict(X_val)
        acc.append(accuracy_score(y_val,y_pred))
        
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        h=0.02
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X_val[:, 0].min() - 1, X_val[:, 0].max() + 1
        y_min, y_max = X_val[:, 1].min() - 1, X_val[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap=cmap_bold,edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification (k = %i, weights = '%s')"% (c, 'uniform'))

        
    fig,ax=plt.subplots()
    ax.plot(c_values,acc,marker='o')
    ax.set_title('Accuracy trend by C-values')
    ax.set_xlabel('C-neighbours number')
    ax.set_ylabel('Accuracy')
    ax.set_xscale('log')
    plt.show()
