import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

career = pd.read_csv("dataset9000.data", header=None)

X = np.array(career.iloc[:, 0:17])  # X is skills
print(X)
y = np.array(career.iloc[:, 17])  # Y is Roles
print(y)

career.columns = ["Web Development", "Infrastructure Automation", "ML Algorithms, AI",
                  "Web Designing Fundamentals", "Blockchain Development", "App Development Skills",
                  "Cloud Infrastructure Management", "Software Testing",
                  "Data Analysis & ML", "Embedded Systems", "Artificial Intelligence", "Cybersecurity Protocols",
                  "AR/VR Design Skills",
                  "Computer Architecture", "Network Fundamentals & Protocols", "Project Planning, Agile, Scrum",
                  "Unity, Unreal Engine, C#", "Roles"]

career.dropna(how="all", inplace=True)
career.head()

scores = {}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=524)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("X_train", X_train)
print("y_train", y_train)

y_pred = knn.predict(X_test)
print("y_pred", y_pred)

scores[5] = metrics.accuracy_score(y_test, y_pred)
print("Accuracy -> ", scores[5] * 100)

pickle.dump(knn, open("careerlast.pkl", "wb"))
print("Knowleadge base pickle file created")
