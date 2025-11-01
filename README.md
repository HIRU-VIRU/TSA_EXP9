### EX.NO: 09  
### A Project on Time Series Analysis on BMW Car Sales Forecasting using ARIMA Model  
**Date:** 01/11/2025

---

### AIM:
To perform time series analysis on BMW car sales data using the ARIMA model in Python and forecast future sales based on historical yearly trends.

### ALGORITHM:

1. **Import Required Libraries**  
   Import pandas, numpy, matplotlib, pmdarima, and statsmodels for data processing, visualization, and modeling.

2. **Load and Preprocess the Dataset**  
   - Read the dataset (`BMW_Car_Sales_Classification.csv`).  
   - Convert the ‘Year’ column into datetime format.  
   - Group sales data by year and compute the total yearly sales.  
   - Sort data by year for time series consistency.

3. **Visualize the Yearly Sales Data**  
   Plot the aggregated yearly sales to observe the overall trend.

4. **Split Data for Training and Testing**  
   Use an 80-20 split to train and validate the model.

5. **Apply Auto ARIMA Model**  
   - Use `auto_arima()` to automatically determine the best parameters (p, d, q).  
   - Set `d=1` to apply differencing for stationarity.

6. **Fit ARIMA Model**  
   Train the ARIMA model using the best (p, d, q) order found.

7. **Forecast and Evaluate**  
   - Forecast sales for the test period.  
   - Calculate Root Mean Squared Error (RMSE) to evaluate accuracy.

8. **Visualize Results**  
   - Plot training, testing, and forecasted sales.  
   - Forecast the next 5 years of sales and plot them.

---

### PROGRAM:
```python
#Name: Hiruthik Sudhakar
#Reg No:212223240054
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# Load dataset
data = pd.read_csv("/content/BMW_Car_Sales_Classification.csv")

# Convert 'Year' to datetime
data['Year'] = pd.to_datetime(data['Year'], format='%Y')

# Aggregate yearly total sales
yearly_sales = data.groupby(data['Year'].dt.year)['Sales_Volume'].sum().reset_index()

# Set proper datetime index
yearly_sales['Year'] = pd.to_datetime(yearly_sales['Year'], format='%Y')
yearly_sales.set_index('Year', inplace=True)

# Sort by year (important for time series)
yearly_sales.sort_index(inplace=True)

# Plot raw yearly data
plt.figure(figsize=(10,5))
plt.plot(yearly_sales['Sales_Volume'], marker='o', color='royalblue')
plt.title('BMW Yearly Sales (Aggregated by Year)', fontsize=14)
plt.xlabel('Year')
plt.ylabel('Total Sales Volume')
plt.grid(alpha=0.4)
plt.tight_layout()
plt.show()

# Train-test split (80-20)
train_size = int(len(yearly_sales) * 0.8)
train = yearly_sales['Sales_Volume'][:train_size]
test = yearly_sales['Sales_Volume'][train_size:]

# Use Auto ARIMA to find the best (p,d,q)
print(" Finding best ARIMA order (with differencing d=1)...\n")

auto_model = auto_arima(
    train,
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    d=1,                    # Force first differencing
    seasonal=False,
    trace=True,
    stepwise=True,
    suppress_warnings=True
)

print("\nBest ARIMA Order Found:", auto_model.order)

# Fit ARIMA with best order
model = ARIMA(train, order=auto_model.order)
fitted_model = model.fit()

# Forecast for the test period
forecast = fitted_model.forecast(steps=len(test))
forecast = pd.Series(forecast.values, index=test.index)

# RMSE Calculation (with NaN protection)
valid_idx = test.dropna().index.intersection(forecast.dropna().index)
if len(valid_idx) > 0:
    rmse = np.sqrt(mean_squared_error(test.loc[valid_idx], forecast.loc[valid_idx]))
    print(f"\n Root Mean Squared Error (RMSE): {rmse:.2f}")
else:
    print("\n Warning: No overlapping samples found for RMSE calculation.")

# Visualization
plt.figure(figsize=(10,5))
plt.plot(train, label='Training Data', color='navy', linewidth=2)
plt.plot(test, label='Testing Data', color='orange', linewidth=2)
plt.plot(forecast, label='ARIMA Forecast', color='green', linestyle='--', linewidth=2)
plt.title(f'BMW Car Sales Forecast using ARIMA{auto_model.order}', fontsize=14)
plt.xlabel('Year')
plt.ylabel('Sales Volume')
plt.legend()
plt.grid(alpha=0.4)
plt.tight_layout()
plt.show()

# Future Forecast (next 5 years)
future_steps = 5
future_forecast = fitted_model.forecast(steps=future_steps)
future_years = pd.date_range(start=yearly_sales.index[-1] + pd.offsets.YearBegin(), periods=future_steps, freq='YS')
future_forecast = pd.Series(future_forecast.values, index=future_years)

plt.figure(figsize=(10,5))
plt.plot(yearly_sales['Sales_Volume'], label='Historical Sales', color='royalblue', linewidth=2)
plt.plot(future_forecast, label='Future Forecast (Next 5 Years)', color='red', linestyle='--', linewidth=2)
plt.title(f'Future BMW Car Sales Forecast (ARIMA{auto_model.order})', fontsize=14)
plt.xlabel('Year')
plt.ylabel('Total Sales Volume')
plt.legend()
plt.grid(alpha=0.4)
plt.tight_layout()
plt.show()
```
### OUTPUT:
**original dataset(aggregated):**
<br>
<img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/af4d17f8-c529-4de2-afd0-353b3c3911fb" />



<br>

**model choosing and rmse:**

<br>
<img width="557" height="281" alt="image" src="https://github.com/user-attachments/assets/0470964c-12ea-4f7b-b1bf-7bfa27a0a539" />
<br>

**forecasting arima:**
<br>
<img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/35cea73a-c7bf-42ca-9ee3-cebee8f22e2d" />
<br>

<img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/15461ef9-683b-47f3-9d37-f7839dd68444" />
<br>

### RESULT:
Thus the program run successfully based on the ARIMA model using python.
