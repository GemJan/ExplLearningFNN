# Thesis Jan Gemander
This repository contains the files for the master thesis "Explanation-based Learning with
Feedforward Neural Networks" for the the [Analytic Computing](https://www.ipvs.uni-stuttgart.de/departments/ac/) department of the Institute for Parallel and Distributed Systems at the University of Stuttgart by Jan Gemander. 

"Explanation-basedLearningFNN_JanGemander.pdf" contains the thesis itself with a signed declaration appended.
The provided titlepage form in 6.2a of the [thesis procedure guidelines](https://www.ipvs.uni-stuttgart.de/departments/ac/teaching/thesis_procedure/) has quite bad formatting when flattening the forms (e.g. printing), 
a cleaned variant which was not used in the thesis is provided as "titlepageCleaned.pdf".
The proposal from the start of the thesis can be found in "Proposal.pdf".
Additionally included are the abtract of our thesis "abstract.txt" and a german abstract "zusammenfassung.txt" in textform.

## annotationScripts
Contains files that were used to annotate our data with arguments. 

Integer file can create any arbitrary new testing data.

We're using given rules from ABML for annotation of Japanese credit and Animal datasets. 

South German credit data set incorporates given [scores](https://data.ub.uni-muenchen.de/23/1/DETAILS.html) that specify if a value is positive or negative to create arguments. 


## data
Contains our datasets including arguments:

Synthtetic Integer dataset

[Japanese Credit](https://archive.ics.uci.edu/ml/datasets/Japanese+Credit+Screening)

[South German Credit](https://archive.ics.uci.edu/ml/datasets/South+German+Credit)

[Animal](https://archive.ics.uci.edu/ml/datasets/zoo)

## implementation
Contains all the files of our implementation and respective test files used to do our experiments:

ContributionFunctions contains functions that we use to compute explanations. 

ExtendedLosses contains our losses that include above contribution functions to optimise towards better explanations.

HelperFunctions contains a few function that help us with managing and evaluating the model.

simpleTest contains a simple test that compares accuracy of multiple losses

extensiveTest includes multiple variations, including averaging over multiple network runs and evaluating explanations. This function was used in a similar form to compute most results of our thesis.
