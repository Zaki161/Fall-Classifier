from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def run_svc(X_train, X_test, y_train, y_test):
    
     # Ustalanie wag klas (jeśli dane są nierównomierne, np. rzadkie przypadki upadków)
    class_weights = {0: 1, 1: 5}  # Przykład ręcznego ustawienia wag, gdzie klasa 1 (upadki) ma wyższą wagę
    print(class_weights)
    # Tworzenie klasyfikatora SVM z wagami klas
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight=class_weights)  # Dodanie parametrów 'class_weight'
    

    # Tworzenie klasyfikatora SVM
    # svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm.fit(X_train, y_train)

    # Predykcja
    y_pred = svm.predict(X_test)

    # Ocena wyników
    print(classification_report(y_test, y_pred, zero_division=1))

    