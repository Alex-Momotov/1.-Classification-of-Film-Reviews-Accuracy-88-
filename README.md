# Classification of Film Reviews, Accuracy: 88.7%



### The pipeline uses Nearest Centroid and Stochastic Gradient Descent to classify 5,800 IMDB film reviews into positive and negative.



## Requirements

- python 3.X

- nltk

- sklearn

- numpy

- memory_profiler

## Type of data
Film discussions is a textual data, which is an instance of dependency-oriented type of data. This is because there exist implicit inner dependencies of data points such as word order. Text data can also be considered a single long string, making it a univariate discrete sequence data type.

## Data Cleaning
First, we used stratifed sampling to extract 2900 texts from each class. Next we cleaned the data set in the following order: remove all punctuation; tokenisation, stop word removal, stemming. Snowball stemming algorithm and stop word removal reduced the resultant lexicon size from the original 37,000 to just over 15,000. Using the lexicon we constructed a bag-of-words (BOW) model in a form of 5,800 by 15,000 NumPy matrix. During the word stem extraction different word variations get transformed into a single stem attribute which can be viewed a dimensionality reduction method. We then compute term frequency-inverse document frequency for each word and multiply these values with respective columns of each vector. After this, we normalise each document-vector by converting it into its unit vector form. This is achieved by dividing every value in each row by its vector magnitude (length). 

## Classification
In this stage we use nearest centroid and stochastic gradient descent classifers and compare their performance. We also use k-fold cross validation with k = 5 with each performance test and report the mean of the 5 learning experiments. Our nearest centroid (NC) method was tested using cosine similarity as well as L2-norm distance metrics. Given the sample size of 5,800 texts (2900 positive, 2900 negative) the classifer successfully predicts if a film is good or bad with 87.2% accuracy when we use euclidean distance metric. We use the same sample size and cross validation method for our stochastic gradient descent (SGD) classifer. For the parameters we choose `l2' penalty and experiment with the random state initialisation. The parameter affects initialisation of random number generator used by SGD. Testing showed that the classifer performs best when random state is initialised to 46 and results in 88.7% accuracy. Out of 1719 testing samples our SGD classifies 1041 flms as good and remaining 678 as bad. For the sake of an experiment we also conduct separation with our nearest centroid classifier. This results in a separation of 1054 and 665 flms to good and bad classes respectively, which is a very similar division as with SGD. We, then, compare the classifcation of each particular data instance between our two algorithms. With the sample size 5,800 the methods give a case-by-case match percentage of 89.52%

## Analysing fail cases
Our SGD wrongly predicted the outcome for films such as "La La Land", "Freddy vs. Jason" and "Catch Me If You Can". We notice that, to an extent, these film reviews use a rhetoric of negating positive statements. For example the review of "La La Land" contains the sentence "Nice to look at, pleasant but nothing special". The phrase uses highly positive words such as 'nice', 'pleasant' and 'special'. On their own the words may appear to be positive which is possibly why our classifer labelled them as such. In the "Freddy vs. Jason" example the review uses words such as 'boring' and 'uninspiring' which are negative, but also uses 'not epic' which can be considered a heavy weight towards positive class by SGD. From this we can conclude that classifcation of discussions based on single-word attributes is somewhat limited in its potential accuracy. If we were to improve our pipeline, we would need to transform the data in a way that can recognise bigrams to capture negation. A promising methodology to achieve this is the latent semantic analysis classifer which maps word combinations of particular length to newly defined concepts.
