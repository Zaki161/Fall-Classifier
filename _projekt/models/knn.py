from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

def run_knn(X_train, X_test, y_train, y_test, n_neighbors=3):
    
    # Normalizacja danych
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Trenowanie modelu KNN (k=5)
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(X_train, y_train)
    print(f"Jest to knn {n_neighbors}")
    # Predykcja i ocena modelu
    y_pred = knn.predict(X_test)
    print(classification_report(y_test, y_pred))
