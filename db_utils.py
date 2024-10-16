from sqlalchemy import create_engine
import yaml
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load credentials from a YAML file
def load_credentials(file_path='credentials.yaml'):
    """Loads the database credentials from a YAML file."""
    with open(file_path, 'r') as file:
        try:
            credentials = yaml.safe_load(file)
            return credentials
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            return None

# RDSDatabaseConnector Class
class RDSDatabaseConnector:
    def __init__(self, credentials):
        """Initializes the RDSDatabaseConnector with database credentials."""
        self.host = credentials.get('RDS_HOST')
        self.password = credentials.get('RDS_PASSWORD')
        self.user = credentials.get('RDS_USER')
        self.database = credentials.get('RDS_DATABASE')
        self.port = credentials.get('RDS_PORT')
        self.engine = None
        
    def init_engine(self):
        """Initializes the SQLAlchemy engine using the database credentials."""
        try:
            connection_string = f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
            self.engine = create_engine(connection_string)
            print("SQLAlchemy engine initialized successfully.")
        except Exception as e:
            print(f"Error initializing the engine: {e}")
            self.engine = None

    def fetch_customer_activity(self):
        """Extracts data from the customer_activity table and returns it as a Pandas DataFrame."""
        if self.engine:
            try:
                query = "SELECT * FROM customer_activity;"
                df = pd.read_sql(query, self.engine)
                return df
            except Exception as e:
                print(f"Error fetching data from customer_activity: {e}")
                return None
        else:
            print("Engine is not initialized.")
            return None
    
    def save_to_csv(self, dataframe, file_path):
        """Saves the given Pandas DataFrame to a CSV file at the specified path."""
        try:
            dataframe.to_csv(file_path, index=False)
            print(f"Data successfully saved to {file_path}")
        except Exception as e:
            print(f"Error saving data to CSV: {e}")

# Load data from a CSV file
def load_data_from_csv(file_path):
    """Loads data from a CSV file into a Pandas DataFrame, prints its shape and a sample of the data."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data shape: {df.shape}")
        print("Sample of the data:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error loading data from CSV: {e}")
        return None

# DataFrameTransform Class for handling transformations, removing correlated columns, and outlier removal
class DataFrameTransform:
    def __init__(self, df):
        """Initializes the DataFrameTransform class with a DataFrame."""
        self.df = df

    def convert_to_category(self, columns):
        """Converts the specified columns to categorical data types."""
        for column in columns:
            if column in self.df.columns:
                self.df[column] = self.df[column].astype('category')
        print(f"Converted columns {columns} to categorical data types.")

    def convert_to_numeric(self, columns, remove_symbols=None):
        """Converts the specified columns to numeric data types, optionally removing symbols."""
        for column in columns:
            if column in self.df.columns:
                if remove_symbols:
                    self.df[column] = self.df[column].replace(remove_symbols, '', regex=True)
                self.df[column] = pd.to_numeric(self.df[column], errors='coerce')
        print(f"Converted columns {columns} to numeric data types.")

    def convert_to_boolean(self, columns):
        """Converts the specified columns to boolean data types."""
        for column in columns:
            if column in self.df.columns:
                self.df[column] = self.df[column].astype(bool)
        print(f"Converted columns {columns} to boolean data types.")

    def remove_outliers_iqr(self, columns):
        """Removes outliers from the specified columns using the IQR method."""
        for column in columns:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Remove rows where the column value is outside the lower and upper bounds
            outliers_removed = self.df[~((self.df[column] < lower_bound) | (self.df[column] > upper_bound))]
            removed_count = len(self.df) - len(outliers_removed)
            print(f"Removed {removed_count} outliers from column '{column}'")
            self.df = outliers_removed

    def identify_highly_correlated(self, threshold=0.9):
        """Identifies columns that are highly correlated based on the given threshold."""
        # Select only numeric columns
        numeric_df = self.df.select_dtypes(include=['number'])
    
        # Calculate the absolute value of the correlation matrix
        corr_matrix = numeric_df.corr().abs()
    
        # Select the upper triangle of the correlation matrix
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
        # Find columns with correlation greater than the threshold
        correlated_columns = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
        print(f"Highly correlated columns (correlation > {threshold}): {correlated_columns}")
        return correlated_columns


    def remove_correlated_columns(self, correlated_columns):
        """Removes the specified correlated columns from the DataFrame."""
        print(f"Removing columns: {correlated_columns}")
        self.df.drop(columns=correlated_columns, inplace=True)

    def get_cleaned_data(self):
        """Returns the cleaned DataFrame."""
        return self.df

# Plotter Class for visualizing NULL values, correlations, outliers, and skewness
class Plotter:
    def __init__(self, df):
        """Initializes the Plotter class with a DataFrame."""
        self.df = df

    def plot_null_values(self, title="Null Values per Column"):
        """Generates a bar plot showing the number of NULL values in each column."""
        null_counts = self.df.isnull().sum()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=null_counts.index, y=null_counts.values)
        plt.xticks(rotation=90)
        plt.title(title)
        plt.ylabel("Number of NULL Values")
        plt.xlabel("Columns")
        plt.show()

    def plot_boxplot(self, columns, title="Boxplot of Columns to Detect Outliers"):
        """Plots boxplots of specified columns to visualize outliers."""
        plt.figure(figsize=(15, 10))
        for i, column in enumerate(columns, 1):
            plt.subplot(len(columns) // 2 + 1, 2, i)
            sns.boxplot(x=self.df[column])
            plt.title(f"Boxplot of {column}")
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def plot_histogram(self, columns, title="Histogram of Columns"):
        """Plots histograms of specified columns to visualize distributions."""
        plt.figure(figsize=(15, 10))
        for i, column in enumerate(columns, 1):
            plt.subplot(len(columns) // 2 + 1, 2, i)
            sns.histplot(self.df[column], kde=True)
            plt.title(f"Histogram of {column}")
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self, title="Correlation Matrix"):
        """Plots the correlation matrix for numeric columns as a heatmap."""
        # Select only numeric columns
        numeric_df = self.df.select_dtypes(include=['number'])
        
        # Compute the correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Plot the correlation matrix as a heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
        plt.title(title)
        plt.show()

# DataFrameInfo Class for performing EDA tasks
class DataFrameInfo:
    def __init__(self, df):
        """Initializes the DataFrameInfo class with a DataFrame."""
        self.df = df

    def describe_columns(self):
        """Describes all columns in the DataFrame and checks data types."""
        print("DataFrame Column Descriptions:")
        print(self.df.describe(include='all'))
        print("\nData Types of Columns:")
        print(self.df.dtypes)

    def extract_statistics(self):
        """Extracts mean, median, and standard deviation from numeric columns."""
        print("Statistical Summary (Mean, Median, Standard Deviation):")
        for column in self.df.select_dtypes(include=['number']).columns:
            print(f"\nColumn: {column}")
            print(f"Mean: {self.df[column].mean()}")
            print(f"Median: {self.df[column].median()}")
            print(f"Standard Deviation: {self.df[column].std()}")

    def count_distinct_values(self):
        """Counts distinct values in categorical columns."""
        print("Distinct Values Count in Categorical Columns:")
        for column in self.df.select_dtypes(include=['category']).columns:
            print(f"\nColumn: {column}")
            print(self.df[column].value_counts())

    def print_shape(self):
        """Prints the shape of the DataFrame."""
        print(f"DataFrame Shape: {self.df.shape}")

    def count_null_values(self):
        """Generates a count and percentage of NULL values in each column."""
        null_count = self.df.isnull().sum()
        total_rows = len(self.df)
        null_percentage = (null_count / total_rows) * 100
        print("\nNull Values Count and Percentage in Each Column:")
        for column in self.df.columns:
            print(f"{column}: {null_count[column]} NULL values ({null_percentage[column]:.2f}% of total)")

    def correlation_matrix(self):
        """Generates and prints the correlation matrix for numeric columns."""
        print("\nCorrelation Matrix for Numeric Columns:")
        numeric_df = self.df.select_dtypes(include=['number'])
        print(numeric_df.corr())

    def get_summary(self):
        """Provides a high-level summary of the DataFrame."""
        self.print_shape()
        self.describe_columns()
        self.count_null_values()

    def get_numeric_summary(self):
        """Provides statistical values for numeric columns."""
        self.extract_statistics()

# Main script execution block
if __name__ == "__main__":
    # Load credentials from the YAML file
    credentials = load_credentials()

    # Initialize the RDSDatabaseConnector with the credentials
    if credentials:
        db_connector = RDSDatabaseConnector(credentials)
        
        # Initialize the SQLAlchemy engine
        db_connector.init_engine()
        
        # Fetch the customer_activity data
        if db_connector.engine:
            customer_activity_data = db_connector.fetch_customer_activity()
            
            # Save the data to a CSV file
            if customer_activity_data is not None:
                db_connector.save_to_csv(customer_activity_data, 'customer_activity_data.csv')

    # Load the customer activity data from the CSV file
    data = load_data_from_csv('customer_activity_data.csv')

    if data is not None:
        # Step 1: Visualize the correlation matrix
        plotter = Plotter(data)
        plotter.plot_correlation_matrix(title="Correlation Matrix of Customer Activity Data")

        # Step 2: Identify highly correlated columns with a threshold of 0.9
        transformer = DataFrameTransform(data)
        correlated_columns = transformer.identify_highly_correlated(threshold=0.9)

        # Step 3: Remove the highly correlated columns
        transformer.remove_correlated_columns(correlated_columns)

        # Step 4: Get the cleaned data and visualize the new correlation matrix
        cleaned_data = transformer.get_cleaned_data()
        plotter = Plotter(cleaned_data)
        plotter.plot_correlation_matrix(title="Correlation Matrix After Removing Highly Correlated Columns")

        # Save the cleaned data to a CSV file
        cleaned_data.to_csv('customer_activity_data_cleaned_no_correlated.csv', index=False)
        print("Cleaned data saved to 'customer_activity_data_cleaned_no_correlated.csv'")

        # Step 5: Perform EDA using DataFrameInfo
        df_info = DataFrameInfo(cleaned_data)

        # Get a high-level summary
        df_info.get_summary()

        # Extract numeric statistics
        df_info.get_numeric_summary()

        # Count distinct values in categorical columns
        df_info.count_distinct_values()

        # Print correlation matrix for numeric columns
        df_info.correlation_matrix()
