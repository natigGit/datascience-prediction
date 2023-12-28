import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('./PropertiesData.csv')

# Imputation
imputer = SimpleImputer(strategy='mean') # or median
df = pd.DataFrame(imputer.fit_transform(df), columns = df.columns)

# Normalization
scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)

# PCA
pca = PCA(n_components=10) # You can change the number of components
df_pca = pd.DataFrame(pca.fit_transform(df), columns = ['CRIM', 'ZN','INDUS', 'CHAS','NOX', 'RM','AGE', 'DIS','RAD','TAX'])

# Split the data
X = df.drop('CRIM', axis=1)
y = df['CRIM']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Calculate MSE
mse = mean_squared_error(y_test, predictions)
print(f'MSE without PCA: {mse}')

# Repeat the process with PCA-transformed data
X_pca = df_pca
y_pca = y
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y_pca, test_size=0.2, random_state=42)

# Linear Regression with PCA
model_pca = LinearRegression()
model_pca.fit(X_train_pca, y_train_pca)
predictions_pca = model_pca.predict(X_test_pca)

# Calculate MSE with PCA
mse_pca = mean_squared_error(y_test_pca, predictions_pca)
print(f'MSE with PCA: {mse_pca}')


#By summarizing the findings and conclusions, I can communicate the results of the analysis and provide more effective techniques for predicting the crime rate per capita.