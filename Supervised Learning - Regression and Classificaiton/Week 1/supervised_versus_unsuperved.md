# Supervised Learning

## Supervised Learning Part 1:
- Learn X -> Y mappings
- Need examples with correct answers for training
- The more training data the better
- The two main types are regression (prediction) and classification
  - Classic example of regression is Linear Regression
- With regression algorithms, the output range is infinte and continuous. With classification algorithms, it is finite => discrete.
- We want to minimize: $(\hat{Y} - Y)^2$
  - Where $\hat{Y}$ is the predicted value and $Y$ is the true value.
  - We use the square of the difference so the value is always positive. The square is faster to differentiate than absolute value.
  - This is definitely true for prediction problems. I don't remember how this can be adapted to classification problems. I think it could still apply if $Y$ is a vector w/ a probability for each possible outcome, and $\hat{Y}$ shares the same schema.

## Supervised Learning Part 2:
- Classification problems can be binary, or have multiple output categories/classes.
- Class or Category are used interchangably when referring to the outputs of a classification algorithm.
- The classic example algorithm is Logistic Regression.
- In classification problems, the possible outcomes are finite and discrete.
- Think of a 2D plot for cancer prediction w/ the axes of Tumor Size and Age. We plot a Circle or X to show malignant or not. The learning algorithm may have to come up with a boundary line to divide the points, such that one side is classified as malignant and the other benign.
- There are often many input variables for both regression and classification.

<br>

# Unsupervised Learning

## Unsupervised Learning Part 1:
- **We are given data that is not associated with any output label**
- **Find a structure, pattern, or something interesting in the data.**
- Example: Clustering: the algorithm may determine that data can be divided/assigned into two categories/clusters.
  - Example: Google News clusters related news articles together. Perhaps by using shared words in title or article.
- Example: Clustering DNA microarray. A column is someone's DNA and a row is a particular gene. Run a clustering algorithm to group people into similar DNA profiles. We don't tell the algorithm what rules to use to categorize people. The algorithm learns how to divide the data, and into how many groups.
- **A clustering algorithm takes data without labels and automatically tries to divide the data into clusters.**

## Unsupervised Learning Part 2:
- A more formal definition:
  - Data comes with only inputs, X.
  - Algorithm has to find structure in the data.
  - Clustering, Anomaly Detection (important in finance for fraud), Dimensionality Reduction (compress a big data set to a smaller one while losing as little information as possible).

<br>

## Jupyter Notebooks and Lab
- 