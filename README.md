Dataset is downloaded from http://archive.ics.uci.edu/ml/datasets/Adult
Features selected are:
1. Age
2. Workclass
3. Education
4. Country
5. Martial Status Feature
6. Occupation
7. Relationship
8. Race
9. Gender
10. Capital
11. Hours per week working

I have also clubbed some of the values of the features, eg occupation in blue and green collar. (higher and lower class)
You can get through the code to understand it.

There are lot of missing values in features,
1. Workclass
2. Country
So, using features, age, education, race, gender, capital, hours per week working we are going to predict missing values
using GradientBoostingClassifier.




so we used classifers to predict those values using fixed features which
don't have any missing values.


1. GradientBoostingClassifier classifier yielded 86% accuracy.
2. Neural Network yielded an accuracy of 88%