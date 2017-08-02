great thanks to Alan Gray:

agray3.github.io/2016/11/29/Demystifying-Data-Input-to-TensorFlow-for-Deep-Learning.html


Run the project as is for linear model and for CNN to make sure everything is OK.
Then paste your own data and run again.

put data files to data folder:
  > mkdir -p data
  > mv train-00000-of-00001 data
  > mv validation-00000-of-00001 data


If you use Python 3, beware of issues like casting.
Python 2: 6/2 is 3 (int)
Python 3: 6/2 is 3.0 (float)

Casting may be needed for some functions take int parameters.
