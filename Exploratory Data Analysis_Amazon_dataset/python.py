import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

#ste2: load data
df=pd.read_csv("amazon Dataset.csv")
print(df)

#step3: overview of data
print("dataset information")
print(df.info())

print("database statistical summary")
print(df.describe())


#print("missing values")
#print(df.isnull().sum())

print("duplicated values",df['Order_ID'].duplicated().sum())

df_cleaned = df.dropna()
df_cleaned = df_cleaned.dropna()
#print(df)"""

#fill missing values


df['Total_Cost'] = df['Total_Cost'].fillna(df['Total_Cost'].median())
#print(df)
df['Shipping_Delay'] = df['Shipping_Delay'].fillna(df['Shipping_Delay'].median())
df['Emp_ID'] = df['Emp_ID'].fillna(df['Emp_ID'].mode()[0])
df['Customer_Rating'] = df['Customer_Rating'].fillna(df['Customer_Rating'].median())
df['Payment_Method'] = df['Payment_Method'].fillna(df['Payment_Method'].mode()[0])
df['Product_Name'] = df['Product_Name'].fillna(df['Product_Name'].mode()[0])
df['Customer_Email'] = df['Customer_Email'].fillna(df['Customer_Email'].mode()[0])
#print(df.isnull().sum())


#convert date columns into datetime format
# Convert the columns to datetime format using astype()
"""df["Shipping_Date"]=df["Shipping_Date"].astype('datetime64[ns]')
#print(df.info())
df["Purchase_Date"]=df["Purchase_Date"].astype('datetime64[ns]')
#print(df.info())
print(df['Purchase_Date'])
print(df['Shipping_Date'])  """

date_col=['Purchase_Date','Shipping_Date']
for i in date_col:
   df[i]=pd.to_datetime(df[i],dayfirst=True,errors='coerce')

#drop rows with invalid date format
df=df.dropna(subset=date_col)
#print(df['Purchase_Date'])
#print(df['Shipping_Date'])

print(df.columns)
# create new features
df['Shipping_Date_year'] = df['Shipping_Date'].dt.year
df['Shipping_order_month'] = df['Shipping_Date'].dt.month
df['Order_Processing_Time'] = (df['Shipping_Date'] - df['Purchase_Date']).dt.days
#print(df)
df['Customer_Satisfaction'] = np.where(df['Customer_Rating'] >= 4, 'High', 'Low')
print(df.head(10))  # Display first 10 rows
df['Purchase_Month'] = df['Purchase_Date'].dt.strftime('%Y-%m')
print(df.head(10))
df['Order_Type'] = np.where(df['Total_Cost'] > df['Total_Cost'].mean(), 'Expensive', 'Cheap')
print(df.head(10))

# Sales by Country
sales_by_country = df.groupby('Country')['Total_Cost'].sum().sort_values(ascending=False)
print("\nSales by Region:\n", sales_by_country)

# Sales by product category
sales_by_category = df.groupby('Category')['Total_Cost'].sum().sort_values(ascending=False)
print("\nSales by Category:\n", sales_by_category)

# category wise total sales
category_wise_total_sales = df.groupby('Category')['Total_Sales'].sum().sort_values(ascending=False)
print("\n Total Sales by Category:\n", category_wise_total_sales)

#Top-performing products
top_products_state_wise = df.groupby('State')['Product_Name'].sum().sort_values(ascending=False).head(10)
print("\nTop-Performing Products:\n",top_products_state_wise)

#product wise final revenue
product_wise_final_revenue = df.groupby('Product_Name')['Final_Revenue'].sum().sort_values(ascending=False).head(10)
print("\n:product wise final revenue\n",product_wise_final_revenue)

# Group by 'Country' and sum the 'Total_Sales', then use head to select top 5
df_top_countries_data = df.groupby('Country')['Total_Sales'].sum().sort_values(ascending=False).head(5).reset_index()

#Monthly Shipping_Cost
monthly_Shipping_Cost = df.groupby(['Shipping_Date_year', 'Shipping_order_month'])['Shipping_Cost'].sum().reset_index()
print("\nmonthly_Shipping_Cost:\n",monthly_Shipping_Cost)

# Group by year and month to calculate the total profit each month
profit_by_month = df.groupby(df['Shipping_Date'].dt.to_period('M'))['Profit'].sum().reset_index()



print(df.columns)
#print(df)
# Step 6: Univariate Analysis
plt.figure(figsize=(8,5))
sns.countplot(x=df['Payment_Method'], palette="deep")
plt.title('Count of Payment Methods')
plt.xlabel('Payment Method')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

numerical_cols = ['Price', 'Total_Cost', 'Profit']
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, palette="cubehelix")
    plt.title(f"Distribution of {col}")
    plt.show()  # Display the plot


# Step 7: Bivariate Analysis
# Relationship between 'Category' and 'Total_Cost'
plt.figure(figsize=(10, 6))
sns.barplot(x='Payment_Method', y='Total_Cost', data=df, palette="crest", ci=None)
plt.title("Total Cost by Payment Method")
plt.xlabel('Payment Method')
plt.ylabel('Total Cost')
plt.xticks(rotation=45)
plt.show()

#relationship between 'Category' and 'Total_Sales'
plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='Total_Sales', data=df,palette="coolwarm", ci=None)
plt.title("Total Sales by Category")
plt.xlabel('Category')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()

#  Multivariate Analysis
plt.figure(figsize=(10, 6))
sns.boxplot(x='State', y='Total_Sales', hue='Order_Type', data=df)
plt.xticks(rotation=45)
plt.title("Total Sales Distribution by State and Order Type")
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x='Country', y='Total_Sales', data=df_top_countries_data, palette="viridis", ci=None)
plt.title("Total Sales Across Different Countries (Top 5)")
plt.xlabel("Country")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.show()

# Time-Based Analysis
# Group by 'Purchase_Month' for both Total Sales and Final Revenue
"""monthly_data = df.groupby('Purchase_Month')[['Total_Sales', 'Final_Revenue']].sum()

# Stacked Bar Plot for Total Sales and Final Revenue
monthly_data.plot(kind='bar', stacked=True, figsize=(12, 6), color=['skyblue', 'orange'])
plt.title("Stacked Bar Plot: Total Sales and Final Revenue", fontsize=16)
plt.xlabel("Purchase Month", fontsize=12)
plt.ylabel("Amount", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show() """

#Reporting Insights
print("\nTop Insights:")
print("\n  1 Final Revenue of products\n",product_wise_final_revenue )
print("\n 2 Total sales in Top 5 Countries\n",df_top_countries_data )
print("\n 3 sales of product by category \n", sales_by_category)
print("\n 4 Monthly Profit Analysis:\n", profit_by_month)
print("\n  5 Total Sales by Country:\n", sales_by_country)

print("\nTop-Performing Products State-wise:\n", top_products_state_wise)

print("\nMonthly Profit Analysis:\n", profit_by_month)

print("\nCustomer Satisfaction Levels:\n", df['Customer_Satisfaction'].value_counts())

print("\nOrder Processing Time Distribution:\n", df['Order_Processing_Time'].describe())

print("\nNumber of Duplicate Orders Detected:\n", df['Order_ID'].duplicated().sum())

# Monthly Sales Trend
monthly_sales_trend = df.groupby('Purchase_Month')['Total_Sales'].sum().reset_index()
print("\nMonthly Sales Trend:\n", monthly_sales_trend)

# Save Cleaned Data
df.to_csv("amazon Dataset.csv", index=False)
