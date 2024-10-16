Exploratory Data Analysis - Online Shopping in Retail

## Overview

This project provides an in-depth analysis of website traffic, user behavior, and revenue generation. It includes scripts that process website activity data and generate insightful visualizations to help teams optimize marketing strategies, identify revenue sources, and address problematic areas.

### Key Features:
- Analyze which regions generate the most/least revenue.
- Visualize the percentage of purchases by new vs. returning customers.
- Compare sales on weekends vs. weekdays.
- Identify the months that generate the most sales.
- Analyze which traffic types (direct, social, or advertising) contribute the most to sales.
- Visualize website traffic based on operating systems, browsers, and mobile vs. desktop users.
- Identify bounce rates for different traffic types and regions.
- Analyze the effectiveness of marketing strategies based on traffic and sales performance.

## Installation

### Prerequisites:
- Python 3.x
- The following Python libraries:
  - `pandas`
  - `numpy`
  - `sqlalchemy`
  - `matplotlib`
  - `seaborn`
  - `scipy`
  - `pyyaml`

You can install the required packages using pip:


pip install pandas numpy sqlalchemy matplotlib seaborn scipy pyyaml
Clone the Repository
git clone https://github.com/syedab1712/Exploratory-Data-Analysis---Online-Shopping-in-Retail.git

Usage
Load Data: Ensure your credentials.yaml file is configured correctly to connect to your RDS database. The db_utils.py script fetches data from an RDS database using credentials stored in the credentials.yaml file.
Example credentials.yaml structure:

yaml

RDS_HOST: your-rds-host
RDS_PASSWORD: your-rds-password
RDS_USER: your-rds-username
RDS_DATABASE: your-database-name
RDS_PORT: 5432
Run the script to fetch the data and save it as a CSV file:



python db_utils.py
Run the Analysis: After fetching the data, the script performs analyses and generates visualizations, helping the team understand:
Revenue breakdown by region
Purchases by customer type (new vs. returning)
Sales comparison between weekends and weekdays
Sales performance by month
Traffic contribution to sales (direct, social, or advertising)
Operating systems, browsers, and mobile vs. desktop usage
Bounce rates by traffic type and region
Traffic generating the most revenue by region
To run the analysis, execute the following command:



python db_utils.py
File Structure


├── db_utils.py          # Main script containing all analysis and visualizations
├── credentials.yaml     # YAML file for storing database credentials
├── README.md            # Documentation for the project
└── customer_activity_data.csv  # Dataset after extracting data from the RDS database
License
This project is licensed under the MIT License. See the LICENSE file for details.

vbnet


This single `README.md` script provides all necessary information for the project, including installation steps, usage, file structure, and license details, as you requested.

Let me know if you need any further modifications!
