import joblib
import numpy as np

def test_model_loads():
    model = joblib.load("model.pkl")
    assert model is not None
    print("✅ Model loaded successfully")

def test_prediction_shape():
    model = joblib.load("model.pkl")
    # make dummy input (10 samples, 4 features as placeholder)
    X_dummy = np.random.rand(10, model.n_features_in_)
    y_pred = model.predict(X_dummy)
    assert y_pred.shape == (10,)
    print("✅ Prediction shape is correct:", y_pred.shape)

if __name__ == "__main__":
    test_model_loads()
    test_prediction_shape()
