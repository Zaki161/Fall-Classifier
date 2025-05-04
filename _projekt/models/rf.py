from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def run_rf(X_train, X_test, y_train, y_test):
    # Trenowanie modelu z uwzględnieniem wag klas
    clf = RandomForestClassifier(
        n_estimators=300, 
        random_state=42,
        class_weight='balanced'  # <<< DODANE!
    )
    clf.fit(X_train, y_train)

    # Predykcja i ocena
    y_pred = clf.predict(X_test)
    print("Raport RF:")
    print(classification_report(y_test, y_pred))