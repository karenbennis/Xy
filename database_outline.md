When selecting data storage for the NLP machine leaning project, it was necessary to find a solution that provided the large storage capacity required while still meeting the budget constraints of the project. The team decided to go with Google Cloud Storage and Google Cloud SQL as it does not charge for egress between the database and Google Colaboratory.  
 

The yelp.csv file was loaded to a Google Cloud Storage bucket which is where the team will be keeping the .csv files used in the project. A Colab notebook named xy_database_prepper.ipynb has been created to complete the following tasks: 

- Connect to the bucket 

- Load yelp.csv into a PySpark DataFrame 

- Change required data types 

- Transform the DataFrame into multiple DataFrames to match schema.sql 

- Load the DataFrames to the Google Cloud SQL database 
 

The main Colab notebook Project.ipynb interacts with the database in the following ways: 

- Connects to the database using Cloud SQL Proxy 

- Queries the database for required tables 

- Joins tables into a DataFrame  

- Performs data transformation adding a "class" column for the machine learning model 

- Loads class information to the review_class table in the database 

- Applies data to NLP machine learning model 

Location of files:

- yelp.csv : /Resources
- ERD : /erd
- database schema : /schema
- notebook : Copy_Of_Project.ipynb
