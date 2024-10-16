

from sqlalchemy import create_engine
import yaml
import pandas as pd


def load_credentials(file_path='credentials.yaml'):
    """Loads the database credentials from a YAML file."""
    with open(file_path, 'r') as file:
        try:
            credentials = yaml.safe_load(file)
            return credentials
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            return None

# Load credentials from the YAML file
credentials = load_credentials()


import pandas as pd

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
            # Create the SQLAlchemy connection string
            connection_string = f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
            
            # Initialize the SQLAlchemy engine
            self.engine = create_engine(connection_string)
            print("SQLAlchemy engine initialized successfully.")
        
        except Exception as e:
            print(f"Error initializing the engine: {e}")
            self.engine = None

    def fetch_customer_activity(self):
        """Extracts data from the customer_activity table and returns it as a Pandas DataFrame."""
        if self.engine:
            try:
                # Define the SQL query
                query = "SELECT * FROM customer_activity;"
                
                # Execute the query and return the result as a DataFrame
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
            # Save the DataFrame to a CSV file
            dataframe.to_csv(file_path, index=False)
            print(f"Data successfully saved to {file_path}")
        except Exception as e:
            print(f"Error saving data to CSV: {e}")



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

import pandas as pd

def load_data_from_csv(file_path):
    """Loads data from a CSV file into a Pandas DataFrame, prints its shape and a sample of the data."""
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Print the shape of the DataFrame
        print(f"Data shape: {df.shape}")
        
        # Print a sample of the data (first 5 rows)
        print("Sample of the data:")
        print(df.head())
        
        return df
    except Exception as e:
        print(f"Error loading data from CSV: {e}")
        return None

# Load the customer activity data from the CSV file
data = load_data_from_csv('customer_activity_data.csv')
