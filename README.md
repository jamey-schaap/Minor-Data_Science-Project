# Data Science project

## The best model
The best model (shallow feed forward network) can be found at `/out/models/best_model`, named `RawData.9c_Adam_1024_0_#FactorScheduler-factor_0.995-stop_factor_0.00075-base_lr_0.00075#_200_25_32_0.19385148584842682.shallow_fnn.keras`. The model's name follows the following naming convention: `{VERSION}_Adam_{units}_{dropoutRate}_{learningRate}_{epochs}_{patience}_{batchsize}_{least_val_loss}.shallow_fnn.keras`. In this case `#FactorScheduler-factor_0.995-stop_factor_0.00075-base_lr_0.00075#` is the learning rate.

## Datasets
All used and created datasets can be found at `/datasets`. The following datasets were used while preprocessing and merging the datasets:
- `Polity5.xls` - Political data; Provided by the project course of the Data Science minor at Rotterdam University of Applied Sciences (RUAS)
- `IMFInvestmentandCapitalStockDataset2021.xlsx` - Investment data; Downloaded from International Monetary Fund (IMF); `https://www.imf.org/`
- `API_SP.POP.TOTL_DS2_en_excel_v2_5871620.xls` - Population per country; Downloaded from the World Bank; `https://www.worldbank.org/`

## Requirements
Python `3.11.x`<br/>

## Setting up

### Auto setup (Windows)
To set the environment up, start a `Powershell 7 Core` `(PWS)` shell, navigate to the root directory of the project and run the script (`.\simple-setup.ps1`). 

### Manual steps
**Note:** A virtual enviroment can also be setup with other tools like Anaconda, but be sure to specify the Python version as 3.11.

Install virtualenv for Python <br/>
`pip install virtualenv`
<br/>
<br/>
Create a virtual python environment <br/>
`python -m venv .\venv`
<br/>
<br/>
Activate the virtual environment <br/>
Unix: `source env/bin/activate` <br/>
PowerShell (PWSH) (Core): `venv\Scripts\Activate.ps1`<br/>
Command Prompt (CMD): `venv\Scripts\activate.bat`
<br/>
<br/>
Update Pip <br/>
`python -m pip install --upgrade pip`
<br/>
<br/>
Install requirements <br/>
`pip install -r requirements.txt`


## During development
Dependent on which editor/IDE you use you might have to activate the virtual Python environment manually. This can be 
done with: <br/>
Unix: `source env/bin/activate` <br/>
PowerShell (PWSH) (Core): `venv\Scripts\Activate.ps1`<br/>
Command Prompt (CMD): `venv\Scripts\activate.bat`


## Configuration
All configuration is done by the files found at `src\configs\`.

### Versioning
Versioning and the number of labels/classes/targets is controlled through the variable `__amount_of_classes` which can 
be found at `src\configs\data.py` `line 96`. This will control both the amount labels/classes/targets used during the 
data preprocessing & merger and during the training of the models. Currently, this is configured for 9 labels with equal 
partitions.

## How to use the different scripts
To use any of the scripts navigate to the `src/` directory. 

### Data preprocessing & merger
Run the `merge_datasets.py` file. This will process the data and create two datasets (xlsx) containing the preprocessed data:
- `src/datasets/MergedDataset-Dataset-V.RawData.\<VERSION\>c.xlsx` &#8594; The dataset for humans, containing the data that is interesting/of use for the project.  
- `src/datasets/MachineLearning-Dataset-V.RawData.\<VERSION\>c.xlsx` &#8594; The dataset for machine learning models, containing the features and labels, which are encoded where needed. 

### Plotting
Edit the `plotting.py` file with what has to be plotted. Examples:
```python
# Example 1 - Simple invoke, specify the dataframe, columns and the plotting function
# if a description exists, it will be shown as the axis label.
simple_invoke(df, x=Column.DUR, y=Prefix.NORM + Column.RISK, plot_func=gf.plot_kde)

# Example 2 - Manually configure the plotting functions
gf.plot_hist(df["norm_risk"], x_label="Risk factor (0..1)")
```
Then run the `plotting.py` file to plot.

### Machine-/Deep learning
For each Jupyter Notebook (`.ipynb`), run the cells under the chapters `Load & split the dataset` and `Utility functions definitions`.

#### KNN, Logistic Regression, SVM & Random Forest
Open the `machine_learning.ipynb`, here all code with regards to KNN, Logistic Regression, SVM & Random Forest can be found. 
Here the models can be trained, tuned and used to predict and the accuracy and feature importance (Shap) can be viewed.

#### Shallow FNN
Open the `shallow_feed_forward_neural_network.ipynb`, here all code with regards to the training of the shallow FNN models can be found.
Here are cells dedicated to the loading, parameter tuning and hyperparameter tuning as well as cells dedicated to view the difference (observed - predicted) and feature importance.

#### Deep FNN
Open the `deep_feed_forward_neural_network.ipynb`, here all code with regards to the training of the shallow FNN models can be found.
Here are cells dedicated to the loading and parameter tuning as well as cells dedicated to view the difference (observed - predicted) and feature importance.

#### General functionality
An excel file can be created containing the incorrectly predicted rows:
```python
# Example - Random forest
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=500, random_state=42) 
rf_model.fit(x_train, train_labels)

_, y_pred = print_results(rf_model)

from machine_learning.utils import output_incorrectly_predicted_xlsx
output_incorrectly_predicted_xlsx(test_df, y_pred, "rf")  
```

A dataframe containing difference between observed and predicted labels can be created and plotted with:
```python
distribution = get_distribution(test_df, y_pred)
print(distribution)
plot_distribution(distribution)
```

The feature importance can be plotted through the use of the Shap library, this can be done through:
```python
## machine_learning.ipynb ##
_, shap_values = calculate_shap_values(model)
shap.summary_plot(shap_values, x_test, feature_names=feature_names,
                  class_names=RISKCLASSIFICATIONS.get_names())

## deep_feed_forward_neural_network.ipynb    ##
## shallow_feed_forward_neural_network.ipynb ##
explainer = shap.KernelExplainer(model.predict, x_train)
shap_values = explainer.shap_values(shap.sample(x_test, 20), nsamples=100, random_state=41)

feature_names = df.columns.tolist()
feature_names.remove(Column.COUNTRY_RISK)
shap.summary_plot(shap_values, x_test, 
                  feature_names=feature_names,
                  class_names=RISKCLASSIFICATIONS.get_names())
```
