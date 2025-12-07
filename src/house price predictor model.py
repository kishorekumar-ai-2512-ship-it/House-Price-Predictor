from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd


data=pd.read_excel(r'C:\Users\kisho\OneDrive\Desktop\AI Mini Projects\House-Price-Predictor\data\cleaned_house_data.xlsx')
data=data.dropna()
x=data[['total_sqft','bath','size','area_type_Plot  Area']]
y=data['price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)


print("R2_Score ",r2_score(y_test,y_pred))
rmse = mean_squared_error(y_test, y_pred)
print("RMSE:", rmse)


