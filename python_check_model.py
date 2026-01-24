import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

PATH = r"D:\Bit\DevopsML\ML_Pipeline\src\model.joblib"

obj = joblib.load(PATH)
print("TYPE:", type(obj))

print("HAS transform:", hasattr(obj, "transform"))
print("HAS predict:", hasattr(obj, "predict"))

print("IS ColumnTransformer:", isinstance(obj, ColumnTransformer))
print("IS Pipeline:", isinstance(obj, Pipeline))

if isinstance(obj, Pipeline):
    print("PIPELINE STEPS:", list(obj.named_steps.keys()))
    # common names: "preprocessor", "model" or "clf"/"reg"
