# inventory-optimizer

To create an inventory optimizer for a retail supply chain, we'll develop a Python program that uses simple statistical methods to predict optimal stock levels and minimize waste. This basic example will involve operations such as data loading, stock level predictions, order recommendations, and error handling.

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class InventoryOptimizer:
    def __init__(self, historical_data, product_line):
        """
        Initialize the InventoryOptimizer with data and product line information.
        
        :param historical_data: DataFrame containing historical sales data.
        :param product_line: List of product names to include.
        """
        self.historical_data = historical_data
        self.product_line = product_line
        self.models = {}
    
    def prepare_data(self):
        """
        Prepare the data for each product in the product line.
        
        :return: Dictionary containing X (features) and y (target) for each product.
        """
        prepared_data = {}
        for product in self.product_line:
            try:
                data = self.historical_data[self.historical_data['product'] == product]
                data = data.sort_values(by='date')
                X = np.arange(len(data)).reshape(-1, 1)  # Use time index as feature
                y = data['sales'].values
                prepared_data[product] = (X, y)
            except KeyError as e:
                print(f"Error with product {product}: {str(e)}. Please check your historical data.")
        return prepared_data
    
    def train_models(self, prepared_data):
        """
        Train a linear regression model for each product.
        
        :param prepared_data: Dictionary of prepared data for each product.
        """
        for product, (X, y) in prepared_data.items():
            try:
                model = LinearRegression().fit(X, y)
                self.models[product] = model
                print(f"Model trained for product: {product}")
            except Exception as e:
                print(f"Failed to train model for product {product}: {str(e)}")
                
    def predict(self, future_periods=1):
        """
        Predict future sales for each product.
        
        :param future_periods: Number of future time periods to predict.
        :return: Dictionary with predicted stock levels for each product.
        """
        predictions = {}
        for product, model in self.models.items():
            try:
                latest_index = len(self.historical_data[self.historical_data['product'] == product]) - 1
                X_future = np.arange(latest_index + 1, latest_index + 1 + future_periods).reshape(-1, 1)
                predictions[product] = model.predict(X_future)
            except Exception as e:
                print(f"Failed to predict for product {product}: {str(e)}")
        return predictions

    def recommend_orders(self, safety_stock=20):
        """
        Recommend order quantities based on predicted sales and a safety stock.
        
        :param safety_stock: Safety stock level to account for fluctuations.
        :return: Dictionary with recommended order quantities for each product.
        """
        recommendations = {}
        predictions = self.predict()
        for product, predicted in predictions.items():
            try:
                recommended_stock = int(predicted[0] + safety_stock)
                current_stock = self.historical_data[self.historical_data['product'] == product]['stock'].values[-1]
                order_quantity = max(0, recommended_stock - current_stock)
                recommendations[product] = order_quantity
            except Exception as e:
                print(f"Failed to recommend order for product {product}: {str(e)}")
        return recommendations

# Sample Usage
if __name__ == "__main__":
    try:
        # Load a sample dataset (replace this with your actual dataset)
        data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
            'product': ['A'] * 50 + ['B'] * 50,
            'sales': np.random.randint(5, 20, size=100),
            'stock': np.random.randint(10, 50, size=100)
        })

        products = ['A', 'B']  # Define the products you want to optimize for
        optimizer = InventoryOptimizer(data, products)

        prepared_data = optimizer.prepare_data()
        optimizer.train_models(prepared_data)
        print("Predictions:", optimizer.predict())
        print("Recommended Orders:", optimizer.recommend_orders())
    except Exception as e:
        print(f"An error occurred in the main execution: {str(e)}")
```

### Explanation:
- **Data Preparation**: The script begins by preparing sales data for each product. This assumes data is already cleaned and loaded in a DataFrame.
- **Model Training**: It uses linear regression to predict sales based on historical data.
- **Prediction and Recommendation**: Predicted sales figures help calculate recommended stock levels, factoring in safety stock.
- **Error Handling**: Incorporates try-except blocks to handle potential issues at each critical step.

This basic program serves as an example. In a production environment, you'd integrate real sales data, refine predictions using more advanced time-series analysis or machine learning models, and tailor safety stock levels based on business requirements.