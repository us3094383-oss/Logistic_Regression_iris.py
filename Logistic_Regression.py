#Project(4) ((Logistic Regression)(With Loss & Accuracy))

#Imports(made)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,log_loss

#DataSet Configuration
u=load_iris()
X=u.data
y=u.target

#Model Settings
model=LogisticRegression(max_iter=200)

#Splitting The DataSet
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=43)

#Training The Model & Predciting Its Output with Each Class Probability
model.fit(X_train,y_train)
y_prd=model.predict(X_test)
y_prob=model.predict_proba(X_test)
print("Accuracy Obtained During Training:",y_prd[:10])
print("Probability Of Each Class:",y_prob[:10])

#Accuracy Check & Losses Occured
a2=accuracy_score(y_test,y_prd)
loss=log_loss(y_test,y_prob)
print("Accuracy:",a2)
print("Losses:",loss)
