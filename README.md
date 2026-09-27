Week 4 – Predictive Modeling and Optimization in Logistics

Target: Delivery_Time_days

This project uses Python to forecast logistics delivery time and propose a scenario-based transport-mode optimization strategy.

Features

Region, Transport_Mode, Priority, Shipment_Volume_kg, Distance_km

Actual delay and final logistics cost were not used as prediction inputs because they can contain information that is only available after delivery activity has occurred.

Models

Linear Regression

Decision Tree Regression

Random Forest Regression

Random Forest hyperparameter tuning with GridSearchCV

5-fold cross-validation

Metrics

MAE, RMSE and R².

Optimization

For each test shipment, Road/Rail/Air/Sea alternatives are simulated. The recommendation chooses the lowest estimated-cost option that meets a 5-day delivery SLA; if none meets the SLA, the fastest predicted option is selected.

Result

Selected model: Tuned Random Forest
Test MAE: 2.300 days
Test RMSE: 3.460 days
Test R²: 0.956

See the Word report for the complete methodology, charts, results, limitations and recommendations.
