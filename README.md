# Thailand House Prices Predictor: Project Overview
This project uses multiple linear regression to predict the house prices in Thailand
* Create a tool that estamates house prices in Thailand to help home buyers negotiate the prices when they are buying a house
* Scraped 2400 houses from baan.kaidee.com using python and selenium
* Engineered features from text of each house description to quantify the value the sellers put on views, schools, universities, and housing assosication
* Optimized Linear, Lasso, and Random Forest Regression using GridsearchCV to reach the best model
* Built a client facing API using flask
## Code and Resources Used
* **Python Version:** 3.7
* **Packages:** pandas, numpy, matplotlib, seaborn, selenium, flask, json, pickle
* **For Web Framework Requirements:** `pip install -r requirements.txt`
* **Scaper Article:** https://medium.com/@ben.sturm/scraping-house-listing-data-using-selenium-and-beautiful-soup-1cbb94ba9492
* **Flask Productization:** https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2
## Web Scraping
Tweaked the web scraper from the article above to scrape 2400 house postings from baan.kaidee.com. With each job, we got the following:
* House Listing Titles
* House Listing Descriptions
* Addresses
* Areas in Square Yard
* Number of Bedrooms
* Number of Bathrooms
* Prices
* Listing Dates
* Status (New or Second Hand)
## Data Cleaning
After scraping the data. I needed to clean it up so that it was usable for our model. I made the followinng changes and created the following variables:
* Parsed numeric data out of prices
* Made columns for subdistricts, districts, and provinces
* Removed rows without bedrooms or bathrooms
* Transformed listing datas or numerical value
* Made columns for it different keywords were listed in the descriptions:
  * school
  * univeristy
  * airport
  * city
  * housing estate
  * view
* Made a columns for titles and descriptions lengths
## EDA
I looked at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights from the pivot tables.

![alt text](https://github.com/Panasak/Thailand_House_Prices_Predictor/blob/main/data_clean/sactter_plot.png)
![alt text](https://github.com/Panasak/Thailand_House_Prices_Predictor/blob/main/data_clean/heat_plot.png)
![alt text](https://github.com/Panasak/Thailand_House_Prices_Predictor/blob/main/data_clean/box_plot.png)
![alt text](https://github.com/Panasak/Thailand_House_Prices_Predictor/blob/main/data_clean/bar_plot.png)
## Model Building
First I transformed the categorical variables into dummy variables. I also split the data into train and test sets with a test size of 33%
I tired three different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret and outliers aren't particularly bad in for this type of model.
I tried three different models:
* **Multiple Linear Regression** - Baseline for the model
* **Lasso Regression** - Because of the sparse data from the many categorical variables. I thought a normalized regression like lasso would be effective
* **Random Forest** - Again, with the sparsity associated with the data, I thought that this would be a good fit
## Model Performance
The Random Forest model outperformed the other approaches on the test and validation sets
* **Random Forest:** MAE = 4411561.95
* **Linear Regression:** MAE = 441239550501459.25
* **Lasso Regression:** MAE = 6423036.02
## Productization
In this step, I built a flask API endpoint that was hosted on a local webserver by following along with the TDS tutorial in the reference section above. The API endpoint takes in a request with a list of values for a house listing and return an estimated price.





