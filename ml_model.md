# Machine Learning Model

### Preliminary data preprocessing

In the beginning we looked at the distribution of star ratings and noticed that it was unevenly split. As we transformed and ran the data through our models, the accuracy was less than 20%, which was unacceptable since the odds of guessing without any machine learning is 20%. We supposed that perhaps there was not enough data for the model to train on for the 1 star and 2 stars reviews. We then decided to sample the data and pull 1000 rows for each star rating. This number was later increased to 2000 per star since we knew Colab could handle 10,000 rows.

Since our plan was to train binary, 3-category, and 5-category models, we created 3 different DataFrames. The following table describes the 3 DataFrames we created.

|DataFrame|Description|
|---------|-----------|
|Binary| 1-star: Negative(0), 2-stars: Negative(0), 3-stars: Dropped from DataFrame, 4-stars: Positive(1), 5-stars: Positive(1)|
|3-category|1-star: Negative(2), 2-stars: Dropped from DataFrame, 3-stars: Neutral(1), 4-stars: Dropped from DataFrame, 5-stars: Positive(0)|
|5-category|1-star: (3), 2-stars:(4), 3-stars: (2), 4-stars: (0), 5-stars: (1)|

### Prelminary feature engineering and selection (decision process)

As part of transforming the data, we went through the NLP process of tokenizing, removing stop words, hashing the data to fit and transform our DataFrame. As we looked more closely after cleaning the data, we determined punctuation had not been removed and there were approximately 62,000 unique columns representing the unique words. This was affecting performance and we needed to further reduce dimensionality. To do this, we employed latent semantic analysis techniques, including stemming and lemmatization.

### Splitting data into training and testing sets

We have split 80% of the data for the training set and the remaining 20% for the testing set. Our decision to use 80% as opposed to something more modest was based on the idea that there would be enough data for our model to train on. From our research of similar studies, we could see that other people attempting to do similar work did not achieve higher accuracy in prediction of sentiment using NLP.

### Model choice

The top choice for the 5-category classification model was the Neural Network model since there are 5 unique categories. Unfortuntately, at this point in time, it has not been possible to run the Neural Network model in Colab. While we continue to brainstorm ways to run this model, we have run the Naive Bayes and Logisitic Regression models. 

As we continue our project life-cycle, we will strive to find a way to run the Neural Network Model. At least for the time being, we will be able to compare the Naive Bayes and Logistic Regression models for all 3 cleaned DataFrames described above. We chose Naive Bayes because of its reputation for being simple, fast, accurate, reliable, and that it works particularly well with NLP problems. Similarly, Logistic Regression scales nicely between 1 or more predictors. 

So far, the limitations is as follows:

- The Neural Network model requires too much RAM in Colab
- The large dataset we intended to use was proved to be too much for Colab

### Preliminary observations

Based on what we were able to run so far, we have observed that the acurracy decreases as we increase the number of categories (applicable for all models). The large dataset sample seems to include entries that may not be English. Based on the preliminary data exploration, it seems reviewers may not be expressing themselves in writing in a way that clearly represents their sentiment in terms of star rating.
