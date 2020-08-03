# Xy (Group project: NLP to predict Yelp rating sentiment)

## Project Overview
As part of the UofT Data Analytics Bootcamp, the final deliverable was group project in which team members collaborate to synthesize and showcase the various skills learned throughout the intensive 6-month course.

Python is used to clean, prepare, and explore the data, and it is used to complete initial analysis. Python libraries and JavaScript libraries (e.g. D3 and Plotly) are used to create visuals to help tell the data story.

Postgres database integration is used to store the cleaned data. Machine learning is implemented to enhance the topic. JavaScript is used to build a dashboard for presenting the results. Finally, the team prepared and delivered a presentation that walked the class through the project, step by step.

### Schedule
The project was broken down into the following 4 segments (over the course of 4 weeks):

|Week|Segment|Description|
|----|-------|-----------|
|1|Sketch It Out|Decide on the overall project. Select research questions, and build a simple model. Connect the model to a fabricated database, using comma-separated values (CSV) or JavaScript Object Notation (JSON) files, to prototype the idea.|
|2|Build the Pieces|Train the model and build out the database that will be used for the final presentation.|
|3|Plug It In|Connect the final database to the model, continue to train the model, and create the dashboard and presentation.|
|4|Put It All Together|Put the final touches on your model, database, and dashboard. Lastly, create and deliver your final presentation to your class.|

#### Team Xy members
The team is comprised of the following individuals:
- Blake Belnap (https://github.com/blocrunx)
- Helen Ly (https://github.com/Helen-Ly)
- Jasmeer Sangha (https://github.com/JasmeerSangha)
- Karen Bennis (https://github.com/karenbennis)

### Topic
Team Xy has decided to apply natural language processing (NLP) and machine learning techniques for the purpose of identifying sentiment to classify Yelp reviews into binary categories ("positive review" / "negative review") and multiclass categories ("positive" / "neutral" / "negative") and ("1 star" / "2 star" / "3 star" / "4 star" / "5 star") based on text content in the reviews.

#### Reason for topic selection
Of all the topics learned, our group agreed that NLP was of particular interest.

When this topic was covered, the course content covered the ETL process. This project will allow us to see how to further incorporate machine learning models with NLP.

As a group we wanted to select a topic that had universal appeal. We agreed that Yelp reviews would be interesting since everyone has an opinion about services and products!

Originally, we'd hoped to build a model for predicting the exact number of stars based on NLP machine learning; however, we opted to include binary classification as well as multiclass categories where there are 3 classes in addition to 5 classes. The decision to include binary and 3-category classification was due to the model predictions were initially quite low in terms of accuracy when predicting the exact number of stars.

### Research questions
- Can NLP be used to predict sentiment of Yelp reviews?
- Can accuracy of NLP model be improved by using larger datasets?
- Can other models predict Yelp sentiment with better accuracy than Naive Bayes model?

### Data source
#### Raw Dataset (big data)

https://www.kaggle.com/shikhar42/yelps-dataset?select=yelp_review.csv
- The dataset is a very large csv file with 5.25 million rows
- As a group, we believed that our machine learning model's accuracy would increase by using a very large dataset
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
We created a Colab notebook (xy_database_prepper.ipynb) to complete the following tasks:

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
Using the sample dataset, the exploratory analysis and plotting revealed that there was an uneven distribution of reviews across the different star ratings, with most of the set being 4 and 5-star reviews.

Despite the imbalance, we created various plots that looked at review text length, useful votes, cool votes, and funny votes to see if there were patterns that could be observed on the basis of star rating.

We discovered immediately that there did not appear to be much variation between a 1-star review and a 5-star review across all metrics.

In hopes of discovering more compelling patterns that might exist, we decided to draw from the larger dataset to see if this would make a difference. Based on the technical limitations of Colab, we realised that sampling more than 10 thousand rows would not be possible, so we decided to sample 10 thousand rows, ensuring equal representation from each star rating.

Unfortunately, the resampled data showed similar patterns to those observed in the sample dataset.

We concluded that star rating does not seem to relate to any of the factors considered in such a profound way. At a higher level, this demonstrates the need for NLP analysis to predict sentiment.

### Machine Learning Model
#### Data preprocessing
In the beginning we looked at the distribution of star ratings and noticed that it was unevenly split. As we transformed and ran the data through our models, the accuracy was less than 20%, which was unacceptable since the odds of guessing without any machine learning is 20%. We supposed that perhaps there was not enough data for the model to train on for the 1 star and 2 stars reviews. We then decided to sample the data and pull 1000 rows for each star rating. This number was later increased to 2000 per star since we knew Colab could handle 10,000 rows.
Since our plan was to train binary, 3-category, and 5-category models, we created 3 different DataFrames. The following table describes the 3 DataFrames we created.
|DataFrame|Description|
|---------|-----------|
|Binary| 1-star: Negative(0), 2-stars: Negative(0), 3-stars: Dropped from DataFrame, 4-stars: Positive(1), 5-stars: Positive(1)|
|3-category|1-star: Negative(2), 2-stars: Dropped from DataFrame, 3-stars: Neutral(1), 4-stars: Dropped from DataFrame, 5-stars: Positive(0)|
|5-category|1-star: (3), 2-stars:(4), 3-stars: (2), 4-stars: (0), 5-stars: (1)|

#### Feature engineering and selection (decision process)
As part of transforming the data, we went through the NLP process of tokenizing, removing stop words, hashing the data to fit and transform our DataFrame. As we looked more closely after cleaning the data, we determined punctuation had not been removed and there were approximately 62,000 unique columns representing the unique words.

This was affecting performance and we needed to further reduce dimensionality. To do this, we employed latent semantic analysis techniques, including stemming and lemmatization.

#### Splitting data into training and testing sets
We split 80% of the data for the training set and the remaining 20% for the testing set. Our decision to use 80% as opposed to something more modest was based on the idea that there would be enough data for our model to train on.

From our research of similar studies, we could see that other people attempting to do similar work did not achieve higher accuracy in prediction of sentiment using NLP.

#### Model choice
The top choice for the 5-category classification model was the Neural Network model since there are 5 unique categories. Unfortuntately, at this point in time, it has not been possible to run the Neural Network model in Colab. While we continue to brainstorm ways to run this model, we have run the Naive Bayes and Logisitic Regression models.

As we continued throughout project life-cycle, we strived to find a way to run the Neural Network Model. Unfortunately, we were not successful.

We were, however, able to compare the Naive Bayes and Logistic Regression models for all 3 cleaned DataFrames described above.

We chose Naive Bayes because of its reputation for being simple, fast, accurate, reliable, and that it works particularly well with NLP problems. Similarly, Logistic Regression scales nicely between 1 or more predictors.

The limitations were as follows:
- The Neural Network model requires too much RAM in Colab
- The large dataset we intended to use was proved to be too much for Colab

#### Observations
Based on what we were able to run so far, we have observed that the acurracy decreases as we increase the number of categories (applicable for all models). The large dataset sample seems to include entries that may not be English. Based on the preliminary data exploration, it seems reviewers may not be expressing themselves in writing in a way that clearly represents their sentiment in terms of star rating.

### Communication Protocols

#### Communication technologies
- Slack
- Whatsapp
- Zoom
- GitHub

#### Communication strategies
Our team collaborates 100% remotely. The COVID 19 pandemic has created a set of circumstances such that we are not able to meet in person at any point in time during this project. From the get-go, we knew that organization and iron clad communication protocols would be paramount to our success as a team.

We are operating as a pseudo-agile team (since we have no product owner). Each week we swap roles in terms of deliverables preparation; however, we are all deeply involved in group discussions on every decision for every deliverable.

##### Verbal communication
We are using the following group chats to communicate in writing:
- Slack (private channel)
- Whatsapp (private group, as contingency for Slack)

We are using Zoom meetings for "in-person" meetings. Most key decisions are made in the Zoom meetings. As well, we divide up the tasks during these meetings.

##### Project tracking
To track everything in an intuitive way, tasks are added to the project board on GitHub.

###### Process: Unassigned tasks (i.e. collaborative group tasks)
1. Create task card on GitHub project board under "To Do".
2. When work on task begins, move task card to "In Progress".
3. When work is completed, move task card to "Done".

Note: Tasks that are not assigned to anyone in particular are not converted to issues.

###### Process Assigned tasks (i.e. individual tasks)
1. Create task card on GitHub project board under "To Do".
2. Convert task to an issue.
3. Assign the issue to the team member.
4. When work on task begins, move task card to "In Progress".
5. Create a branch with the same name as the task to create traceability between task and branch. (This also lets the whole team know where to find this work in the repository without having to ask).
6. Complete the task and commit work to the branch.
7. Create a pull request to merge with the master branch.
8. All team members review the pull request and approve.
9. Merge with master branch.
10. When work is completed (merged to master branch), GitHub will automatically move task card to "Done".

Note: We implemented a 3-person review requirement for merging with the master branch in hopes of preventing merge conflicts. This also ensures that everyone on the team has line of sight as to what is pushed to the master branch.