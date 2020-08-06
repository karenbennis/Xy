# Xy (Group project: NLP to predict Yelp rating sentiment)

## Project Overview
As part of the UofT Data Analytics Bootcamp, the final deliverable was group project in which team members collaborate to synthesize and showcase the various skills learned throughout the intensive 6-month course.

Postgres database integration is used to store the cleaned data. Python is used to clean, prepare, and explore the data, and it is used to complete analysis. Machine learning together with NLP is implemented to gain insights that were not observable from categorical features exisitng in the raw data. Plotly dash was used to create visuals to help tell the data story on our dashboard. Finally, the team prepared and delivered a presentation that walked the class through the project, step by step.

### Schedule
The project was broken down into the following 4 segments (over the course of 4 weeks):

|Week|Segment|Description|
|----|-------|-----------|
|1|Sketch It Out|Decide on the overall project. Select research questions, and build a simple model. Connect the model to a fabricated database, using comma-separated values (CSV) or JavaScript Object Notation (JSON) files, to prototype the idea.|
|2|Build the Pieces|Train the model and build out the database that will be used for the final presentation.|
|3|Plug It In|Connect the final database to the model, continue to train the model, and create the dashboard and presentation.|
|4|Put It All Together|Put the final touches on your model, database, and dashboard. Lastly, create and deliver your final presentation to your class.|

### Team Xy members
The team is comprised of the following individuals:
- Blake Belnap (https://github.com/blocrunx)
- Helen Ly (https://github.com/Helen-Ly)
- Jasmeer Sangha (https://github.com/JasmeerSangha)
- Karen Bennis (https://github.com/karenbennis)

### Topic
Team Xy applied natural language processing (NLP) and machine learning techniques for the purpose of identifying sentiment to classify Yelp reviews into the following binary and multiclass classifications based on text content in the reviews:
- binary ("positive review" / "negative review")
- 3-category ("positive" / "neutral" / "negative")
- 5-category ("1 star" / "2 star" / "3 star" / "4 star" / "5 star")

#### Reason for topic selection
Of all the topics learned in the Data Bootcamp, our group agreed that NLP was of particular interest.

When this topic was presented, the course content covered the ETL process. This project allowed us to  further incorporate machine learning models with NLP.

As a group we wanted to select a topic that had universal appeal. We agreed that Yelp reviews would be interesting due to its universal appeal.

Originally, we'd hoped to build a neural network machine learning model that predicts the exact number of stars based on NLP machine learning; however, we opted to include binary classification as well as multiclass categories where there are 3 classes in addition to 5 classes. The decision to include binary and 3-category classification was due to the model predictions were initially quite low in terms of accuracy when predicting the exact number of stars.

### Research questions
- Can NLP be used to predict sentiment of Yelp reviews?
- Can accuracy of NLP model be improved by using larger datasets?

### Resources
- Technologies / Tools: Google Cloud Storage, Google Colaboratory, PostgreSQL, PgAdmin, Plotly-dash
- Languages: Python, CSS
- Libraries (Colab): pandas, nltk.stem, string, re, pyspark
- Libraries (dashboard): dash, dash_core_components, dash_html_components, dash_bootstrap_components, dash.dependencies, requests, base64, io, plotly, collections, pandas, sklearn, numpy

### Data source
#### Raw Dataset (big data)

https://www.kaggle.com/shikhar42/yelps-dataset?select=yelp_review.csv
- The dataset is a very large csv file with 5.25 million rows
- As a group, we believed that our machine learning model's accuracy would increase by using a very large dataset
- This dataset had an extra column (by comparison with the preliminary dataset, described below) which was dropped prior to analysis
- Technological limitations with regard to running Colab prevented us from including more data, as initially intended

#### Raw dataset (preliminary dataset)

https://www.kaggle.com/omkarsabnis/yelp-reviews-dataset
- This dataset is a smaller csv file with 10 thousand rows
- The dataset has same columns as large dataset, which makes for good sample data for testing our initial NLP model
- This dataset does not have an equal distribution of reviews corresponding with each star rating

### Database setup

#### Considerations
With regard to the data storage for this project, we felt it was necessary to find a solution that was able to accommodate large storage capacity while still meeting the budget constraints of the project.

Since Google Cloud Storage and Google Cloud SQL allow free egress between the database and Google Colaboratory, we made the decision to store the data files in a Google Could Storage bucket.

#### Setup
We created a Colab notebook to complete the following tasks:

1. Connect to the Google Cloud Storage bucket.
2. Load the data file into a PySpark DataFrame.
3. Change the required data types.
4. Transform the DataFrame into multiple DataFrames to match the schema.
5. Load the DataFrames to the Google Cloud SQL database.

#### Accessing the data
The main Colab notebook Project.ipynb interacts with the database in the following ways:
1. Connects to the database using Cloud SQL Proxy.
2. Query the database for required tables.
3. Join tables into a DataFrame.
4. Perform data transformation adding a "class" column for the machine learning model.
5. Load class information to the review_class table in the database.


### Exploratory analysis phase
Using the sample dataset, the exploratory analysis and plotting revealed that there was an uneven distribution of reviews across the different star ratings, with most of the set being 4 and 5-star reviews, as shown below:

![](https://github.com/karenbennis/Xy/blob/master/Visuals/distribution.png)

Despite the imbalance, we created various plots that looked at review text length, useful votes, cool votes, and funny votes to see if there were patterns that could be observed on the basis of star rating.

We discovered immediately that there did not appear to be much variation between any star review (1, 2, 3, 4, or 5-stars) across all metrics as shown in the plots below:

![](https://github.com/karenbennis/Xy/blob/master/Visuals/Facet.png)
![](https://github.com/karenbennis/Xy/blob/master/Visuals/Boxplot.png)
![](https://github.com/karenbennis/Xy/blob/master/Visuals/Violin.png)

In hopes of discovering more compelling patterns that might exist, we decided to draw from the larger dataset to see if this would make a difference. Based on the technical limitations of Colab, we realised that sampling more than 10 thousand rows would not be possible, so we decided to sample 10 thousand rows, ensuring equal representation from each star rating.

![](https://github.com/karenbennis/Xy/blob/master/Visuals/distribution-balanced.png)

Unfortunately, similar patterns to those observed in the sample dataset were observed in the resampled data, as shown below.

![](https://github.com/karenbennis/Xy/blob/master/Visuals/Facet-balanced.png)
![](https://github.com/karenbennis/Xy/blob/master/Visuals/Boxplot-balanced.png)
![](https://github.com/karenbennis/Xy/blob/master/Visuals/Violin-balanced.png

We concluded that star rating does not seem to relate to any of the factors considered in such a profound way. At a higher level, this demonstrates the need for NLP analysis to predict sentiment.

### Machine Learning Model
#### Data preprocessing
In the beginning we looked at the distribution of star ratings and noticed that it was unevenly split. As we transformed and ran the data through our models, the accuracy was less than 20%, which was unacceptable since the odds of guessing without any machine learning is 20%. We supposed that perhaps there was not enough data for the model to train on for the 1 star and 2 stars reviews. We then decided to sample the data and pull 1000 rows for each star rating. This number was later increased to 2000 per star rating, because we knew Colab could handle 10,000 rows.
Since our plan was to train binary, 3-category, and 5-category models, we created 3 different DataFrames. The following table describes the 3 DataFrames we created, each of which is represented in a different Colab notebook.

|DataFrame|Description|
|---------|-----------|
|Binary| 1-star: Negative(0), 2-stars: Negative(0), 3-stars: Dropped from DataFrame, 4-stars: Positive(1), 5-stars: Positive(1)|
|3-category|1-star: Negative(2), 2-stars: Dropped from DataFrame, 3-stars: Neutral(1), 4-stars: Dropped from DataFrame, 5-stars: Positive(0)|
|5-category|1-star: (3), 2-stars:(4), 3-stars: (2), 4-stars: (0), 5-stars: (1)|

#### Feature engineering and selection (decision process)
As part of transforming the data, we went through the NLP process of tokenizing, removing stop words, hashing the data to fit and transform our DataFrame. As we looked more closely after cleaning the data, we determined punctuation had not been removed and there were approximately 62,000 unique columns representing the unique words.

This was affecting performance, and we needed to further reduce dimensionality. To do this, we employed latent semantic analysis techniques (stemming).

#### Splitting data into training and testing sets
We split 80% of the data for the training set and the remaining 20% for the testing set. Our decision to use 80% as opposed to something more modest was based on the idea that there would be enough data for our model to train on.

From our research of similar studies, we could see that other projects attempting to do similar work did not achieve higher accuracy in prediction of sentiment using NLP.

#### Model choice
The top choice for the 5-category classification model was the Neural Network model since there are 5 unique categories. Unfortuntately, at this point in time, it has not been possible to run the Neural Network model in Colab. While we continued to brainstorm ways to run this model, we ran the Naive Bayes and Logisitic Regression models.

As we continued throughout the project life-cycle, we strived to find a way to run the Neural Network Model. Unfortunately, we were not successful.

We were, however, able to compare the Naive Bayes and Logistic Regression models for all 3 cleaned DataFrames described above.

We chose Naive Bayes because of its reputation for being simple, fast, accurate, reliable, and because it works particularly well for NLP problems. Similarly, Logistic Regression scales nicely between 1 or more predictors.

### Observations
We have observed that the acurracy decreases as we increase the number of categories (applicable for all models). The large dataset sample seems to include entries that may not be English.

Based on this data exploration, it seems reviewers may not be expressing themselves in writing in a way that clearly represents their sentiment in terms of star rating.

The limitations were as follows:
- The Neural Network model requires too much RAM in Colab
- The large dataset we intended to use was proved to be too much for Colab

### Lessons learned
Despite the benefits of Google Cloud Service, Google Colab was a bottleneck, and ultimately unable to perform what we intended. Furthermore, the models which did work also needed to be coded using sci-kit learn for the dashboard (as opposed to pyspark).

We also learned that dense layers of neural networks use much more RAM than other ML models when using sparse matrices for hashed text.

Had there been more time, We may have been able to indentify cost-effective solutions that could actually handle the big data and run a neural network machine learning model.

### Recommendations for further analysis
To expand upon the current analysis, we recommend the following:
- Train and test machine learning models on a larger dataset
- Find a viable solution for running the neural network machine learning model
- After running the neural network machine learning model successfully, apply the model to different datasets (e.g. amazon reviews, IMDb reviews, Google reviews) to see its effectiveness in predicting sentiment when applied to different datasets