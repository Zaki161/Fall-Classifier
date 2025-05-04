from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

def run_mlp(X_train, X_test, y_train, y_test):   
   
    # Tworzenie klasyfikatora MLP
    mlp = MLPClassifier(hidden_layer_sizes=(50,50), max_iter=1000)
    mlp.fit(X_train, y_train)

    # Predykcja
    y_pred = mlp.predict(X_test)

    # Ocena wyników
    print(classification_report(y_test, y_pred))