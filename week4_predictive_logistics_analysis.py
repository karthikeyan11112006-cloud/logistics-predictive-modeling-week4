import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("week3_logistics_dataset.csv")
features = ["Region","Transport_Mode","Priority","Shipment_Volume_kg","Distance_km"]
target = "Delivery_Time_days"
X, y = df[features], df[target]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)

cat = ["Region","Transport_Mode","Priority"]
num = ["Shipment_Volume_kg","Distance_km"]
prep = ColumnTransformer([
    ("cat",OneHotEncoder(handle_unknown="ignore"),cat),
    ("num",StandardScaler(),num)
])

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5,min_samples_leaf=4,random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=250,max_depth=8,min_samples_leaf=2,random_state=42)
}

for name, model in models.items():
    pipe = Pipeline([("prep",prep),("model",model)])
    pipe.fit(X_train,y_train)
    pred = pipe.predict(X_test)
    print(name)
    print("MAE:", mean_absolute_error(y_test,pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test,pred)))
    print("R2:", r2_score(y_test,pred))

# Hyperparameter tuning
rf = Pipeline([("prep",prep),("model",RandomForestRegressor(random_state=42))])
grid = GridSearchCV(
    rf,
    {"model__n_estimators":[100,250],
     "model__max_depth":[5,8,None],
     "model__min_samples_leaf":[1,2,4]},
    cv=KFold(5,shuffle=True,random_state=42),
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)
grid.fit(X_train,y_train)
best_model = grid.best_estimator_
print("Best parameters:",grid.best_params_)

# Optimization concept:
# For each shipment, simulate Road, Rail, Air and Sea.
# Predict delivery time for each mode and estimate cost.
# Select the lowest-cost mode that meets a 5-day SLA.
