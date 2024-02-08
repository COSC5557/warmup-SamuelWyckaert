In this assignement, I used two models to determine the wine quality :
    - linear regression
    - KNN for classification

I took a baseline that choose randomely the wine quality, it gave me an accuracy around 14%.
The linear regression model obtain an accuracy of 28.6%.
For the KNN model, I fluctuate the value of K betwwen 1 and 250. Surprinsgly, the model is the best with a K = 1 (see plot.png to see the plot).


#### Sources :
Regression : 
- https://realpython.com/linear-regression-in-python/
Classification : 
- https://www.tutorialspoint.com/how-to-implement-linear-classification-with-python-scikit-learn
Baseline :
- https://medium.com/@preethiprakash29/understanding-baseline-models-in-machine-learning-3ed94f03d645#:~:text=Creating%20a%20baseline%20model%20helps,of%20the%20imbalanced%20class%20issue.