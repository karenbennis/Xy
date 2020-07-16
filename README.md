# Xy (Group project: NLP to predict Yelp rating sentiment)

## Project Overview
As part of the UofT Data Analytics Bootcamp, the final deliverable is group project in which team members collaborate to synthesize and showcase the various skills learned throughout this intensive 6-month course.

Python is used to clean, prepare, and explore the data, and it is used to complete initial analysis. Python libraries and JavaScript libraries (e.g. D3 and Plotly) are used to create visuals to help tell your data story.

Postgres database integration is used to store the cleaned data. Machine learning is implemented to enhance the topic. JavaScript is used to build a dashboard for presenting the results. Finally, the team prepared and delivered a presentation that walked the class through the project, step by step.

### Schedule
The project is broken down into the following 4 segments (over the course of 4 weeks):

|Week|Segment|Description|
|----|-------|-----------|
|1|Sketch It Out|Decide on the overall project. Select research questions, and build a simple model. Connect the model to a fabricated database, using comma-separated values (CSV) or JavaScript Object Notation (JSON) files, to prototype the idea.|
|2|Build the Pieces|Train the model and build out the database that will be used for the final presentation.|
|3|Plug It In|Connect the final database to the model, continue to train the model, and create the dashboard and presentation.|
|4|Put It All Together|Put the final touches on your model, database, and dashboard. Lastly, create and deliver your final presentation to your class.|

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

## Communication Protocols

### Team Xy members
The team is comprised of the following individuals:
- Blake Belnap (https://github.com/blocrunx)
- Helen Ly (https://github.com/Helen-Ly)
- Jasmeer Sangha (https://github.com/JasmeerSangha)
- Karen Bennis (https://github.com/karenbennis)

### Communication technologies
- Slack
- Whatsapp
- Zoom
- GitHub

### Communication strategies
Our team collaborates 100% remotely. The COVID 19 pandemic has created a set of circumstances such that we are not able to meet in person at any point in time during this project. From the get-go, we knew that organization and iron clad communication protocols would be paramount to our success as a team.

We are operating as a pseudo-agile team (since we have no product owner). Each week we swap roles in terms of deliverables preparation; however, we are all deeply involved in group discussions on every decision for every deliverable.

#### Verbal communication
We are using the following group chats to communicate in writing:
- Slack (private channel)
- Whatsapp (private group, as contingency for Slack)

We are using Zoom meetings for "in-person" meetings. Most key decisions are made in the Zoom meetings. As well, we divide up the tasks during these meetings.

#### Project tracking
To track everything in an intuitive way, tasks are added to the project board on GitHub.

##### Process: Unassigned tasks (i.e. collaborative group tasks)
1. Create task card on GitHub project board under "To Do".
2. When work on task begins, move task card to "In Progress".
3. When work is completed, move task card to "Done".

Note: Tasks that are not assigned to anyone in particular are not converted to issues.

##### Process Assigned tasks (i.e. individual tasks)
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

Note: We have implemented a 3-person review requirement for merging with the master branch in hopes of preventing merge conflicts. This also ensures that everyone on the team has line of sight as to what is pushed to the master branch.