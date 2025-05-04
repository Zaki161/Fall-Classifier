from config import window_size, shift_size, test_split
from loader import load_data
from extraction import extract_features_full
from splits import osoby_v2,osoby,losowy ### podział danych
from models.knn import run_knn
from models.rf import run_rf
from models.svc import run_svc
from models.mlp import run_mlp

def main():
    df = load_data()
    features_df = extract_features_full(df, window_size, shift_size)


    ## trenowanie przez oddzielenie osob
    # X_train, X_test, y_train, y_test = osoby_v2(features_df, test_split)
    X_train, X_test, y_train, y_test = losowy(features_df)

    run_knn(X_train, X_test, y_train, y_test)
    run_rf(X_train, X_test, y_train, y_test)
    run_svc(X_train, X_test, y_train, y_test)
    run_mlp(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()