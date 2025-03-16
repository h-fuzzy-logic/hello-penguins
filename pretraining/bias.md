# Data bias
Before training a machine learning model, the data must be evaluated for its suitability. Below are some questions and answers for the penguins dataset. 

## Where was this data collected?
From three islands in Palmer Archipelago, Antarctica:
* Biscoe
* Dream
* Torgersen

*Location matters because this data set focuses on a specific location in Antarctica and on only three species of penguins. Other regions and species are not represented in this dataset.*  

## When was this data collected?
It was collected from 2007-2009

*Timeframe matters because penguin population and characteristics change over time. This dataset is a snapshot in time.* 

## How was this data collected?
The measurements are in millimeters and grams. 

*Depending on who made the measurement and which equipment was used, there could be slight differences in how the measurements were taken. Since the collection time frame spans two years, different equipment could have been used.*  

## Are the penguin species represented equally?
The dataset represents 344 penguins:
* 152 Adelie
* 124 Gentoo
* 68 Chinstrap

*If the dataset is imbalanced because it has more of one species than the others, the results of machine learning models may be impacted.*

*When a model is trained with biased data, the model may not perform well for the underrepresented groups.* 

