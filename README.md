# Xy (Group project: NLP to predict Yelp star ratings)

## Project Overview
As part of the UofT Data Analytics Bootcamp, the final deliverable is group project in which team members collaborate to synthesize and showcase the various skills learned throughout this intensive 6-month course.

Python is used to clean, prepare, and explore the data, and it is used to complete initial analysis. Python libraries and JavaScript libraries (e.g. D3 and Plotly) are used to create visuals to help tell your data story.

Postgres database integration is used to store the cleaned data. Machine learning is implemented to enhance the topic. JavaScript is used to build a dashboard for presenting the results. Finally, the team prepared and delivered a presentation that walked the class through the project, step by step.

The project is broken down into the following 4 segments (over the course of 4 weeks):

|Week|Segment|Description|
|----|-------|-----------|
|1|Sketch It Out|Decide on the overall project. Select research questions, and build a simple model. Connect the model to a fabricated database, using comma-separated values (CSV) or JavaScript Object Notation (JSON) files, to prototype the idea.|
|2|Build the Pieces|Train the model and build out the database that will be used for the final presentation.|
|3|Plug It In|Connect the final database to the model, continue to train the model, and create the dashboard and presentation.|
|4|Put It All Together|Put the final touches on your model, database, and dashboard. Lastly, create and deliver your final presentation to your class.|

The team is comprised of the following individuals:
- Blake Belnap (https://github.com/blocrunx)
- Helen Ly (https://github.com/Helen-Ly)
- Jasmeer Sangha (https://github.com/JasmeerSangha)
- Karen Bennis (https://github.com/karenbennis)

### Topic
Team Xy has decided to apply natural language processing (NLP) and machine learning techniques for the purpose of identifying sentiment to classify Yelp reviews into binary categories ("positive review" / "negative review") based on text content in the reviews.

### Reason for topic selection
Of all the topics learned, our group agreed that NLP was of particular interest.

When this topic was covered, the course content covered the ETL process. This project will allow us to see how further incorporate machine learning models with NLP.

As a group we wanted to select a topic that had universal appeal. We agreed that restaurant reviews would be interesting since everyone has an opinion about food!

Originally, we'd hoped to build a model for predicting the exact number of stars based on NLP machine learning; however, we opted for binary classification instead, in the interest of managing our time effectively.

### Description of data source
#### Raw Dataset (big data)

https://www.kaggle.com/shikhar42/yelps-dataset?select=yelp_review.csv
- The dataset is a very large csv file with 5.25 million rows
- As a group, we believe that our machine learning model's accuracy will increase by using a very large dataset

#### Raw dataset (preliminary dataset)

https://www.kaggle.com/omkarsabnis/yelp-reviews-dataset/kernels
- This dataset is a smaller csv file with 10 thousand rows
- The dataset has same columns as large dataset, which makes for good sample data for testing our initial NLP model

### Research questions
- Can NLP be used to predict sentiment of Yelp reviews?
- Can accuracy of NLP model be improved by using larger datasets?
- Can other models predict Yelp sentiment with better accuracy than Naive Bayes model?