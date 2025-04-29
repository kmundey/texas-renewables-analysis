# **Texas Renewable Generation Project**
*Final project repository for I 320M*

## **I. Data Sources**
- Electric Reliability Council of Texas (ERCOT): - Historical renewable power generation data (wind and solar)
- European Centre for Medium-Range Weather Forecasts (ERA5): Historical wind-related weather conditions
- National Solar Radiation Data Base (NSRDB): Historical solar-related weather conditions data

## **III. Short-Term Wind Power Forecasting for ERCOT: A Convolutional Neural Network Approach**
*Katherine & Kayla*

[Presentation Slides](https://docs.google.com/presentation/d/1TK-kIXhdPX2I4-8dliGmBrolojeLi4EfbT6sDtHG04s/edit?usp=sharing)

[Report](https://docs.google.com/document/d/18foUTwQ0_6GvCXHzqPOnypqigv-xFI-WK5xJTTFsrlE/edit?usp=sharing)

Texas is the leading producer of wind energy in the United States, generating approximately 40,652 MW in 2023, which represented 28.6% of Texas’ total energy mix. Texas’s wind energy sector continues to grow rapidly, with total capacity increasing from around 10,000 megawatts in 2011 to over 40,000 megawatts in 2022. Texas wind plays a vital role in meeting the state’s increasing electricity demands and supporting economic development. As wind becomes an even larger share of Texas’s energy mix, understanding and analyzing wind patterns and generation trends is crucial to ensure reliable, efficient, and sustainable energy delivery. 

Components: 
- Data Analysis and Visualization
- Deep Learning Wind Generation Forecasting Model

**Deep Learning to Forecast Wind Generation: 1D CNN**
The `wind_analysis` folder contains a deep learning model built to forecast hourly wind power generation (MW) for ERCOT using historical wind generation data from ERCOT and weather conditions data from ERA5. The forecasted output predicts the next hour of wind power generation based on time-series data spanning from January 1, 2021, to December 31, 2023.

**Key Features**
- *Data Preprocessing:* The dataset is cleaned and transformed to handle missing values, normalize features, and generate new interaction features (e.g., time of day, wind generation lags, rolling statistics) to improve model performance.
- *Model:* A 1D Convolutional Neural Network (CNN) is employed to capture temporal dependencies and patterns in wind power generation data. The CNN architecture is optimized to work with time-series data, specifically utilizing a sliding window approach to forecast the next hour's wind generation. Utilized TensorFlow and Keras. 
- *Cross-Validation:* To prevent overfitting and ensure robust model performance, the model undergoes cross-validation using TimeSeriesSplit, a technique that splits the data while maintaining the temporal order, ensuring the model is tested on unseen data at each fold. Utlized Sci-Kit Learn. 

**Model Architecture**
- The model utilizes 1D Convolutional Layers to learn short-term temporal patterns in wind generation data.
- MaxPooling is applied to reduce dimensionality and improve generalization.
- Fully connected layers (Dense) are used for decision-making based on learned features, and Dropout layers are included to prevent overfitting.

**Output**
- The model forecasts the next hour (total of 24 hours) of wind power generation, providing a reliable prediction for grid operators, energy providers, and other stakeholders in the renewable energy industry.

### IV. Tovi & Ruhama: Solar Generation
...
