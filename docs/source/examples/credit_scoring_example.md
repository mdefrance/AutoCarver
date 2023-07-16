# Settings

## Installation


```python
!pip install --upgrade autocarver
```

    Requirement already satisfied: autocarver in c:\users\defra\.conda\envs\py39\lib\site-packages (5.1.2)
    Requirement already satisfied: ipython in c:\users\defra\.conda\envs\py39\lib\site-packages (from autocarver) (8.11.0)
    Requirement already satisfied: numpy in c:\users\defra\.conda\envs\py39\lib\site-packages (from autocarver) (1.24.2)
    Requirement already satisfied: tqdm in c:\users\defra\.conda\envs\py39\lib\site-packages (from autocarver) (4.65.0)
    Requirement already satisfied: scipy in c:\users\defra\.conda\envs\py39\lib\site-packages (from autocarver) (1.10.1)
    Requirement already satisfied: statsmodels in c:\users\defra\.conda\envs\py39\lib\site-packages (from autocarver) (0.14.0)
    Requirement already satisfied: pandas in c:\users\defra\.conda\envs\py39\lib\site-packages (from autocarver) (1.5.3)
    Requirement already satisfied: scikit-learn in c:\users\defra\.conda\envs\py39\lib\site-packages (from autocarver) (1.2.2)
    Requirement already satisfied: decorator in c:\users\defra\.conda\envs\py39\lib\site-packages (from ipython->autocarver) (5.1.1)
    Requirement already satisfied: jedi>=0.16 in c:\users\defra\.conda\envs\py39\lib\site-packages (from ipython->autocarver) (0.18.2)
    Requirement already satisfied: traitlets>=5 in c:\users\defra\.conda\envs\py39\lib\site-packages (from ipython->autocarver) (5.9.0)
    Requirement already satisfied: stack-data in c:\users\defra\.conda\envs\py39\lib\site-packages (from ipython->autocarver) (0.6.2)
    Requirement already satisfied: matplotlib-inline in c:\users\defra\.conda\envs\py39\lib\site-packages (from ipython->autocarver) (0.1.6)
    Requirement already satisfied: colorama in c:\users\defra\.conda\envs\py39\lib\site-packages (from ipython->autocarver) (0.4.6)
    Requirement already satisfied: backcall in c:\users\defra\.conda\envs\py39\lib\site-packages (from ipython->autocarver) (0.2.0)
    Requirement already satisfied: pickleshare in c:\users\defra\.conda\envs\py39\lib\site-packages (from ipython->autocarver) (0.7.5)
    Requirement already satisfied: pygments>=2.4.0 in c:\users\defra\.conda\envs\py39\lib\site-packages (from ipython->autocarver) (2.14.0)
    Requirement already satisfied: prompt-toolkit!=3.0.37,<3.1.0,>=3.0.30 in c:\users\defra\.conda\envs\py39\lib\site-packages (from ipython->autocarver) (3.0.38)
    Requirement already satisfied: pytz>=2020.1 in c:\users\defra\.conda\envs\py39\lib\site-packages (from pandas->autocarver) (2022.7.1)
    Requirement already satisfied: python-dateutil>=2.8.1 in c:\users\defra\.conda\envs\py39\lib\site-packages (from pandas->autocarver) (2.8.2)
    Requirement already satisfied: joblib>=1.1.1 in c:\users\defra\.conda\envs\py39\lib\site-packages (from scikit-learn->autocarver) (1.2.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\defra\.conda\envs\py39\lib\site-packages (from scikit-learn->autocarver) (3.1.0)
    Requirement already satisfied: patsy>=0.5.2 in c:\users\defra\.conda\envs\py39\lib\site-packages (from statsmodels->autocarver) (0.5.3)
    Requirement already satisfied: packaging>=21.3 in c:\users\defra\.conda\envs\py39\lib\site-packages (from statsmodels->autocarver) (23.0)
    Requirement already satisfied: parso<0.9.0,>=0.8.0 in c:\users\defra\.conda\envs\py39\lib\site-packages (from jedi>=0.16->ipython->autocarver) (0.8.3)
    Requirement already satisfied: six in c:\users\defra\.conda\envs\py39\lib\site-packages (from patsy>=0.5.2->statsmodels->autocarver) (1.16.0)
    Requirement already satisfied: wcwidth in c:\users\defra\.conda\envs\py39\lib\site-packages (from prompt-toolkit!=3.0.37,<3.1.0,>=3.0.30->ipython->autocarver) (0.2.6)
    Requirement already satisfied: pure-eval in c:\users\defra\.conda\envs\py39\lib\site-packages (from stack-data->ipython->autocarver) (0.2.2)
    Requirement already satisfied: asttokens>=2.1.0 in c:\users\defra\.conda\envs\py39\lib\site-packages (from stack-data->ipython->autocarver) (2.2.1)
    Requirement already satisfied: executing>=1.2.0 in c:\users\defra\.conda\envs\py39\lib\site-packages (from stack-data->ipython->autocarver) (1.2.0)
    

## Setting up samples

This dataset can be found from the corresponding Kaggle competition at https://www.kaggle.com/competitions/GiveMeSomeCredit/


```python
import pandas as pd

data_path = "GiveMeSomeCredit"

credit_data = pd.read_csv(f"{data_path}/cs-training.csv", index_col=0)
print(credit_data.shape)
credit_data.head()
```

    (150000, 11)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SeriousDlqin2yrs</th>
      <th>RevolvingUtilizationOfUnsecuredLines</th>
      <th>age</th>
      <th>NumberOfTime30-59DaysPastDueNotWorse</th>
      <th>DebtRatio</th>
      <th>MonthlyIncome</th>
      <th>NumberOfOpenCreditLinesAndLoans</th>
      <th>NumberOfTimes90DaysLate</th>
      <th>NumberRealEstateLoansOrLines</th>
      <th>NumberOfTime60-89DaysPastDueNotWorse</th>
      <th>NumberOfDependents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.766127</td>
      <td>45</td>
      <td>2</td>
      <td>0.802982</td>
      <td>9120.0</td>
      <td>13</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.957151</td>
      <td>40</td>
      <td>0</td>
      <td>0.121876</td>
      <td>2600.0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0.658180</td>
      <td>38</td>
      <td>1</td>
      <td>0.085113</td>
      <td>3042.0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0.233810</td>
      <td>30</td>
      <td>0</td>
      <td>0.036050</td>
      <td>3300.0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0.907239</td>
      <td>49</td>
      <td>1</td>
      <td>0.024926</td>
      <td>63588.0</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import train_test_split

X_train, X_dev = train_test_split(credit_data, test_size=0.33, random_state=42)
```

## Picking up columns to Carve


```python
X_train.dtypes
```




    SeriousDlqin2yrs                          int64
    RevolvingUtilizationOfUnsecuredLines    float64
    age                                       int64
    NumberOfTime30-59DaysPastDueNotWorse      int64
    DebtRatio                               float64
    MonthlyIncome                           float64
    NumberOfOpenCreditLinesAndLoans           int64
    NumberOfTimes90DaysLate                   int64
    NumberRealEstateLoansOrLines              int64
    NumberOfTime60-89DaysPastDueNotWorse      int64
    NumberOfDependents                      float64
    dtype: object




```python
X_train.isna().mean()
```




    SeriousDlqin2yrs                        0.000000
    RevolvingUtilizationOfUnsecuredLines    0.000000
    age                                     0.000000
    NumberOfTime30-59DaysPastDueNotWorse    0.000000
    DebtRatio                               0.000000
    MonthlyIncome                           0.197383
    NumberOfOpenCreditLinesAndLoans         0.000000
    NumberOfTimes90DaysLate                 0.000000
    NumberRealEstateLoansOrLines            0.000000
    NumberOfTime60-89DaysPastDueNotWorse    0.000000
    NumberOfDependents                      0.026129
    dtype: float64




```python
target = "SeriousDlqin2yrs"
quantitative_features = [feature for feature in X_train if feature != target]
```

# Feature processing with AutoCarver
## Fitting train samples and testint robustness


```python
from AutoCarver import AutoCarver

auto_carver = AutoCarver(
    quantitative_features=quantitative_features,
    qualitative_features=[],
    sort_by='cramerv',  # Best combination according to Cramer's V
    dropna=False,  # don't want to groups nans with other values, leave that to XGBoost 
    min_freq=0.1,  # minimum frequency per modality
    max_n_mod=5,  # maximum number of modality per carved feature
    copy=True,  # in order not to modify X_train directly
    pretty_print=True,  # prints nice tables
)
x_discretized = auto_carver.fit_transform(
    # specifying dataset to carve
    X_train, X_train[target],
    # specifying a dataset to test robustness
    X_dev=X_dev, y_dev=X_dev[target]
)
```

    ------
    [Discretizer] Fit Quantitative Features
    ---
     - [QuantileDiscretizer] Fit ['age', 'NumberOfDependents', 'DebtRatio', 'RevolvingUtilizationOfUnsecuredLines', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfOpenCreditLinesAndLoans', 'NumberRealEstateLoansOrLines', 'NumberOfTime30-59DaysPastDueNotWorse', 'MonthlyIncome']
     - [BaseDiscretizer] Transform Quantitative ['age', 'NumberOfDependents', 'DebtRatio', 'RevolvingUtilizationOfUnsecuredLines', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfOpenCreditLinesAndLoans', 'NumberRealEstateLoansOrLines', 'NumberOfTime30-59DaysPastDueNotWorse', 'MonthlyIncome']
     - [OrdinalDiscretizer] Fit ['NumberOfDependents', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'RevolvingUtilizationOfUnsecuredLines', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfOpenCreditLinesAndLoans', 'NumberRealEstateLoansOrLines', 'age', 'MonthlyIncome']
    ------
    
     - [BaseDiscretizer] Transform Quantitative ['NumberOfDependents', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'RevolvingUtilizationOfUnsecuredLines', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfOpenCreditLinesAndLoans', 'NumberRealEstateLoansOrLines', 'age', 'MonthlyIncome']
     - [BaseDiscretizer] Transform Quantitative ['NumberOfDependents', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'RevolvingUtilizationOfUnsecuredLines', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfOpenCreditLinesAndLoans', 'NumberRealEstateLoansOrLines', 'age', 'MonthlyIncome']
    
    ------
    [AutoCarver] Fit NumberOfDependents (1/10)
    ---
    
     - [AutoCarver] Raw feature distribution
    


<style type="text/css">
#T_4a7df_row0_col0 {
  background-color: #a7c5fe;
  color: #000000;
}
#T_4a7df_row0_col1, #T_4a7df_row2_col0 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_4a7df_row1_col0 {
  background-color: #f7b599;
  color: #000000;
}
#T_4a7df_row1_col1 {
  background-color: #93b5fe;
  color: #000000;
}
#T_4a7df_row2_col1 {
  background-color: #afcafc;
  color: #000000;
}
#T_4a7df_row3_col0, #T_4a7df_row3_col1 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
</style>
<table id="T_4a7df" style='display:inline'>
  <caption>X distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_4a7df_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_4a7df_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
    <tr>
      <th class="index_name level0" >NumberOfDependents</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_4a7df_level0_row0" class="row_heading level0 row0" >x <= 0.0</th>
      <td id="T_4a7df_row0_col0" class="data row0 col0" >0.060000</td>
      <td id="T_4a7df_row0_col1" class="data row0 col1" >0.580000</td>
    </tr>
    <tr>
      <th id="T_4a7df_level0_row1" class="row_heading level0 row1" >0.0 < x <= 1.0</th>
      <td id="T_4a7df_row1_col0" class="data row1 col0" >0.073000</td>
      <td id="T_4a7df_row1_col1" class="data row1 col1" >0.175000</td>
    </tr>
    <tr>
      <th id="T_4a7df_level0_row2" class="row_heading level0 row2" >1.0 < x</th>
      <td id="T_4a7df_row2_col0" class="data row2 col0" >0.085000</td>
      <td id="T_4a7df_row2_col1" class="data row2 col1" >0.219000</td>
    </tr>
    <tr>
      <th id="T_4a7df_level0_row3" class="row_heading level0 row3" >__NAN__</th>
      <td id="T_4a7df_row3_col0" class="data row3 col0" >0.048000</td>
      <td id="T_4a7df_row3_col1" class="data row3 col1" >0.026000</td>
    </tr>
  </tbody>
</table>
    <style type="text/css">
#T_ccc36_row0_col0 {
  background-color: #adc9fd;
  color: #000000;
}
#T_ccc36_row0_col1, #T_ccc36_row2_col0 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_ccc36_row1_col0 {
  background-color: #f6a283;
  color: #000000;
}
#T_ccc36_row1_col1 {
  background-color: #94b6ff;
  color: #000000;
}
#T_ccc36_row2_col1 {
  background-color: #aec9fc;
  color: #000000;
}
#T_ccc36_row3_col0, #T_ccc36_row3_col1 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
</style>
<table id="T_ccc36" style='display:inline'>
  <caption>X_dev distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_ccc36_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_ccc36_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
    <tr>
      <th class="index_name level0" >NumberOfDependents</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_ccc36_level0_row0" class="row_heading level0 row0" >x <= 0.0</th>
      <td id="T_ccc36_row0_col0" class="data row0 col0" >0.057000</td>
      <td id="T_ccc36_row0_col1" class="data row0 col1" >0.579000</td>
    </tr>
    <tr>
      <th id="T_ccc36_level0_row1" class="row_heading level0 row1" >0.0 < x <= 1.0</th>
      <td id="T_ccc36_row1_col0" class="data row1 col0" >0.074000</td>
      <td id="T_ccc36_row1_col1" class="data row1 col1" >0.177000</td>
    </tr>
    <tr>
      <th id="T_ccc36_level0_row2" class="row_heading level0 row2" >1.0 < x</th>
      <td id="T_ccc36_row2_col0" class="data row2 col0" >0.086000</td>
      <td id="T_ccc36_row2_col1" class="data row2 col1" >0.218000</td>
    </tr>
    <tr>
      <th id="T_ccc36_level0_row3" class="row_heading level0 row3" >__NAN__</th>
      <td id="T_ccc36_row3_col0" class="data row3 col0" >0.042000</td>
      <td id="T_ccc36_row3_col1" class="data row3 col1" >0.026000</td>
    </tr>
  </tbody>
</table>



    Grouping modalities   : 100%|██████████| 3/3 [00:00<?, ?it/s]
    Computing associations: 100%|██████████| 3/3 [00:00<00:00, 3078.76it/s]
    Testing robustness    :   0%|          | 0/3 [00:00<?, ?it/s]

    
     - [AutoCarver] Carved feature distribution
    

    
    


<style type="text/css">
#T_aa5dd_row0_col0 {
  background-color: #a7c5fe;
  color: #000000;
}
#T_aa5dd_row0_col1, #T_aa5dd_row2_col0 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_aa5dd_row1_col0 {
  background-color: #f7b599;
  color: #000000;
}
#T_aa5dd_row1_col1 {
  background-color: #93b5fe;
  color: #000000;
}
#T_aa5dd_row2_col1 {
  background-color: #afcafc;
  color: #000000;
}
#T_aa5dd_row3_col0, #T_aa5dd_row3_col1 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
</style>
<table id="T_aa5dd" style='display:inline'>
  <caption>X distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_aa5dd_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_aa5dd_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_aa5dd_level0_row0" class="row_heading level0 row0" >x <= 0.0</th>
      <td id="T_aa5dd_row0_col0" class="data row0 col0" >0.060000</td>
      <td id="T_aa5dd_row0_col1" class="data row0 col1" >0.580000</td>
    </tr>
    <tr>
      <th id="T_aa5dd_level0_row1" class="row_heading level0 row1" >0.0 < x <= 1.0</th>
      <td id="T_aa5dd_row1_col0" class="data row1 col0" >0.073000</td>
      <td id="T_aa5dd_row1_col1" class="data row1 col1" >0.175000</td>
    </tr>
    <tr>
      <th id="T_aa5dd_level0_row2" class="row_heading level0 row2" >1.0 < x</th>
      <td id="T_aa5dd_row2_col0" class="data row2 col0" >0.085000</td>
      <td id="T_aa5dd_row2_col1" class="data row2 col1" >0.219000</td>
    </tr>
    <tr>
      <th id="T_aa5dd_level0_row3" class="row_heading level0 row3" >__NAN__</th>
      <td id="T_aa5dd_row3_col0" class="data row3 col0" >0.048000</td>
      <td id="T_aa5dd_row3_col1" class="data row3 col1" >0.026000</td>
    </tr>
  </tbody>
</table>
    <style type="text/css">
#T_b8e41_row0_col0 {
  background-color: #adc9fd;
  color: #000000;
}
#T_b8e41_row0_col1, #T_b8e41_row2_col0 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_b8e41_row1_col0 {
  background-color: #f6a283;
  color: #000000;
}
#T_b8e41_row1_col1 {
  background-color: #94b6ff;
  color: #000000;
}
#T_b8e41_row2_col1 {
  background-color: #aec9fc;
  color: #000000;
}
#T_b8e41_row3_col0, #T_b8e41_row3_col1 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
</style>
<table id="T_b8e41" style='display:inline'>
  <caption>X_dev distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_b8e41_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_b8e41_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_b8e41_level0_row0" class="row_heading level0 row0" >x <= 0.0</th>
      <td id="T_b8e41_row0_col0" class="data row0 col0" >0.057000</td>
      <td id="T_b8e41_row0_col1" class="data row0 col1" >0.579000</td>
    </tr>
    <tr>
      <th id="T_b8e41_level0_row1" class="row_heading level0 row1" >0.0 < x <= 1.0</th>
      <td id="T_b8e41_row1_col0" class="data row1 col0" >0.074000</td>
      <td id="T_b8e41_row1_col1" class="data row1 col1" >0.177000</td>
    </tr>
    <tr>
      <th id="T_b8e41_level0_row2" class="row_heading level0 row2" >1.0 < x</th>
      <td id="T_b8e41_row2_col0" class="data row2 col0" >0.086000</td>
      <td id="T_b8e41_row2_col1" class="data row2 col1" >0.218000</td>
    </tr>
    <tr>
      <th id="T_b8e41_level0_row3" class="row_heading level0 row3" >__NAN__</th>
      <td id="T_b8e41_row3_col0" class="data row3 col0" >0.042000</td>
      <td id="T_b8e41_row3_col1" class="data row3 col1" >0.026000</td>
    </tr>
  </tbody>
</table>



    ------
    
    
    ------
    [AutoCarver] Fit NumberOfTime30-59DaysPastDueNotWorse (2/10)
    ---
    
     - [AutoCarver] Raw feature distribution
    


<style type="text/css">
#T_afc31_row0_col0, #T_afc31_row1_col1 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_afc31_row0_col1, #T_afc31_row1_col0 {
  background-color: #b40426;
  color: #f1f1f1;
}
</style>
<table id="T_afc31" style='display:inline'>
  <caption>X distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_afc31_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_afc31_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
    <tr>
      <th class="index_name level0" >NumberOfTime30-59DaysPastDueNotWorse</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_afc31_level0_row0" class="row_heading level0 row0" >x <= 0.0</th>
      <td id="T_afc31_row0_col0" class="data row0 col0" >0.040000</td>
      <td id="T_afc31_row0_col1" class="data row0 col1" >0.839000</td>
    </tr>
    <tr>
      <th id="T_afc31_level0_row1" class="row_heading level0 row1" >0.0 < x</th>
      <td id="T_afc31_row1_col0" class="data row1 col0" >0.208000</td>
      <td id="T_afc31_row1_col1" class="data row1 col1" >0.161000</td>
    </tr>
  </tbody>
</table>
    <style type="text/css">
#T_39418_row0_col0, #T_39418_row1_col1 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_39418_row0_col1, #T_39418_row1_col0 {
  background-color: #b40426;
  color: #f1f1f1;
}
</style>
<table id="T_39418" style='display:inline'>
  <caption>X_dev distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_39418_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_39418_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
    <tr>
      <th class="index_name level0" >NumberOfTime30-59DaysPastDueNotWorse</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_39418_level0_row0" class="row_heading level0 row0" >x <= 0.0</th>
      <td id="T_39418_row0_col0" class="data row0 col0" >0.039000</td>
      <td id="T_39418_row0_col1" class="data row0 col1" >0.842000</td>
    </tr>
    <tr>
      <th id="T_39418_level0_row1" class="row_heading level0 row1" >0.0 < x</th>
      <td id="T_39418_row1_col0" class="data row1 col0" >0.207000</td>
      <td id="T_39418_row1_col1" class="data row1 col1" >0.158000</td>
    </tr>
  </tbody>
</table>



    Grouping modalities   : 100%|██████████| 1/1 [00:00<?, ?it/s]
    Computing associations: 100%|██████████| 1/1 [00:00<00:00, 1001.98it/s]
    Testing robustness    :   0%|          | 0/1 [00:00<?, ?it/s]

    
     - [AutoCarver] Carved feature distribution
    

    
    


<style type="text/css">
#T_e52b7_row0_col0, #T_e52b7_row1_col1 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_e52b7_row0_col1, #T_e52b7_row1_col0 {
  background-color: #b40426;
  color: #f1f1f1;
}
</style>
<table id="T_e52b7" style='display:inline'>
  <caption>X distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_e52b7_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_e52b7_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_e52b7_level0_row0" class="row_heading level0 row0" >x <= 0.0</th>
      <td id="T_e52b7_row0_col0" class="data row0 col0" >0.040000</td>
      <td id="T_e52b7_row0_col1" class="data row0 col1" >0.839000</td>
    </tr>
    <tr>
      <th id="T_e52b7_level0_row1" class="row_heading level0 row1" >0.0 < x</th>
      <td id="T_e52b7_row1_col0" class="data row1 col0" >0.208000</td>
      <td id="T_e52b7_row1_col1" class="data row1 col1" >0.161000</td>
    </tr>
  </tbody>
</table>
    <style type="text/css">
#T_28345_row0_col0, #T_28345_row1_col1 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_28345_row0_col1, #T_28345_row1_col0 {
  background-color: #b40426;
  color: #f1f1f1;
}
</style>
<table id="T_28345" style='display:inline'>
  <caption>X_dev distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_28345_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_28345_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_28345_level0_row0" class="row_heading level0 row0" >x <= 0.0</th>
      <td id="T_28345_row0_col0" class="data row0 col0" >0.039000</td>
      <td id="T_28345_row0_col1" class="data row0 col1" >0.842000</td>
    </tr>
    <tr>
      <th id="T_28345_level0_row1" class="row_heading level0 row1" >0.0 < x</th>
      <td id="T_28345_row1_col0" class="data row1 col0" >0.207000</td>
      <td id="T_28345_row1_col1" class="data row1 col1" >0.158000</td>
    </tr>
  </tbody>
</table>



    ------
    
    
    ------
    [AutoCarver] Fit DebtRatio (3/10)
    ---
    
     - [AutoCarver] Raw feature distribution
    


<style type="text/css">
#T_7b1a9_row0_col0, #T_7b1a9_row0_col1, #T_7b1a9_row1_col1, #T_7b1a9_row2_col1, #T_7b1a9_row3_col1, #T_7b1a9_row5_col1, #T_7b1a9_row6_col1 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_7b1a9_row1_col0 {
  background-color: #9bbcff;
  color: #000000;
}
#T_7b1a9_row2_col0, #T_7b1a9_row4_col0 {
  background-color: #688aef;
  color: #f1f1f1;
}
#T_7b1a9_row3_col0 {
  background-color: #445acc;
  color: #f1f1f1;
}
#T_7b1a9_row4_col1, #T_7b1a9_row6_col0, #T_7b1a9_row7_col1 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_7b1a9_row5_col0 {
  background-color: #efcfbf;
  color: #000000;
}
#T_7b1a9_row7_col0 {
  background-color: #4e68d8;
  color: #f1f1f1;
}
</style>
<table id="T_7b1a9" style='display:inline'>
  <caption>X distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_7b1a9_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_7b1a9_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
    <tr>
      <th class="index_name level0" >DebtRatio</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_7b1a9_level0_row0" class="row_heading level0 row0" >x <= 31.1m</th>
      <td id="T_7b1a9_row0_col0" class="data row0 col0" >0.052000</td>
      <td id="T_7b1a9_row0_col1" class="data row0 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_7b1a9_level0_row1" class="row_heading level0 row1" >31.1m < x <= 133.6m</th>
      <td id="T_7b1a9_row1_col0" class="data row1 col0" >0.070000</td>
      <td id="T_7b1a9_row1_col1" class="data row1 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_7b1a9_level0_row2" class="row_heading level0 row2" >133.6m < x <= 213.9m</th>
      <td id="T_7b1a9_row2_col0" class="data row2 col0" >0.061000</td>
      <td id="T_7b1a9_row2_col1" class="data row2 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_7b1a9_level0_row3" class="row_heading level0 row3" >213.9m < x <= 287.6m</th>
      <td id="T_7b1a9_row3_col0" class="data row3 col0" >0.054000</td>
      <td id="T_7b1a9_row3_col1" class="data row3 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_7b1a9_level0_row4" class="row_heading level0 row4" >287.6m < x <= 467.4m</th>
      <td id="T_7b1a9_row4_col0" class="data row4 col0" >0.061000</td>
      <td id="T_7b1a9_row4_col1" class="data row4 col1" >0.200000</td>
    </tr>
    <tr>
      <th id="T_7b1a9_level0_row5" class="row_heading level0 row5" >467.4m < x <= 648.0m</th>
      <td id="T_7b1a9_row5_col0" class="data row5 col0" >0.088000</td>
      <td id="T_7b1a9_row5_col1" class="data row5 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_7b1a9_level0_row6" class="row_heading level0 row6" >648.0m < x <= 3.8</th>
      <td id="T_7b1a9_row6_col0" class="data row6 col0" >0.114000</td>
      <td id="T_7b1a9_row6_col1" class="data row6 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_7b1a9_level0_row7" class="row_heading level0 row7" >3.8 < x</th>
      <td id="T_7b1a9_row7_col0" class="data row7 col0" >0.056000</td>
      <td id="T_7b1a9_row7_col1" class="data row7 col1" >0.200000</td>
    </tr>
  </tbody>
</table>
    <style type="text/css">
#T_9061d_row0_col0 {
  background-color: #445acc;
  color: #f1f1f1;
}
#T_9061d_row0_col1, #T_9061d_row2_col1, #T_9061d_row3_col0 {
  background-color: #3f53c6;
  color: #f1f1f1;
}
#T_9061d_row1_col0 {
  background-color: #6e90f2;
  color: #f1f1f1;
}
#T_9061d_row1_col1, #T_9061d_row5_col1, #T_9061d_row7_col0 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_9061d_row2_col0 {
  background-color: #4e68d8;
  color: #f1f1f1;
}
#T_9061d_row3_col1, #T_9061d_row6_col1 {
  background-color: #3d50c3;
  color: #f1f1f1;
}
#T_9061d_row4_col0 {
  background-color: #6384eb;
  color: #f1f1f1;
}
#T_9061d_row4_col1 {
  background-color: #be242e;
  color: #f1f1f1;
}
#T_9061d_row5_col0 {
  background-color: #b9d0f9;
  color: #000000;
}
#T_9061d_row6_col0, #T_9061d_row7_col1 {
  background-color: #b40426;
  color: #f1f1f1;
}
</style>
<table id="T_9061d" style='display:inline'>
  <caption>X_dev distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_9061d_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_9061d_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
    <tr>
      <th class="index_name level0" >DebtRatio</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_9061d_level0_row0" class="row_heading level0 row0" >x <= 31.1m</th>
      <td id="T_9061d_row0_col0" class="data row0 col0" >0.056000</td>
      <td id="T_9061d_row0_col1" class="data row0 col1" >0.101000</td>
    </tr>
    <tr>
      <th id="T_9061d_level0_row1" class="row_heading level0 row1" >31.1m < x <= 133.6m</th>
      <td id="T_9061d_row1_col0" class="data row1 col0" >0.064000</td>
      <td id="T_9061d_row1_col1" class="data row1 col1" >0.099000</td>
    </tr>
    <tr>
      <th id="T_9061d_level0_row2" class="row_heading level0 row2" >133.6m < x <= 213.9m</th>
      <td id="T_9061d_row2_col0" class="data row2 col0" >0.058000</td>
      <td id="T_9061d_row2_col1" class="data row2 col1" >0.101000</td>
    </tr>
    <tr>
      <th id="T_9061d_level0_row3" class="row_heading level0 row3" >213.9m < x <= 287.6m</th>
      <td id="T_9061d_row3_col0" class="data row3 col0" >0.055000</td>
      <td id="T_9061d_row3_col1" class="data row3 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_9061d_level0_row4" class="row_heading level0 row4" >287.6m < x <= 467.4m</th>
      <td id="T_9061d_row4_col0" class="data row4 col0" >0.062000</td>
      <td id="T_9061d_row4_col1" class="data row4 col1" >0.199000</td>
    </tr>
    <tr>
      <th id="T_9061d_level0_row5" class="row_heading level0 row5" >467.4m < x <= 648.0m</th>
      <td id="T_9061d_row5_col0" class="data row5 col0" >0.077000</td>
      <td id="T_9061d_row5_col1" class="data row5 col1" >0.099000</td>
    </tr>
    <tr>
      <th id="T_9061d_level0_row6" class="row_heading level0 row6" >648.0m < x <= 3.8</th>
      <td id="T_9061d_row6_col0" class="data row6 col0" >0.115000</td>
      <td id="T_9061d_row6_col1" class="data row6 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_9061d_level0_row7" class="row_heading level0 row7" >3.8 < x</th>
      <td id="T_9061d_row7_col0" class="data row7 col0" >0.054000</td>
      <td id="T_9061d_row7_col1" class="data row7 col1" >0.202000</td>
    </tr>
  </tbody>
</table>



    Grouping modalities   : 100%|██████████| 98/98 [00:00<00:00, 7372.02it/s]
    Computing associations: 100%|██████████| 98/98 [00:00<00:00, 3828.24it/s]
    Testing robustness    :   2%|▏         | 2/98 [00:00<00:00, 799.98it/s]

    
     - [AutoCarver] Carved feature distribution
    

    
    


<style type="text/css">
#T_c48d0_row0_col0 {
  background-color: #4a63d3;
  color: #f1f1f1;
}
#T_c48d0_row0_col1, #T_c48d0_row3_col0 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_c48d0_row1_col0 {
  background-color: #5572df;
  color: #f1f1f1;
}
#T_c48d0_row1_col1, #T_c48d0_row4_col1 {
  background-color: #aac7fd;
  color: #000000;
}
#T_c48d0_row2_col0 {
  background-color: #ead5c9;
  color: #000000;
}
#T_c48d0_row2_col1, #T_c48d0_row3_col1, #T_c48d0_row4_col0 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
</style>
<table id="T_c48d0" style='display:inline'>
  <caption>X distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_c48d0_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_c48d0_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_c48d0_level0_row0" class="row_heading level0 row0" >x <= 31.1m</th>
      <td id="T_c48d0_row0_col0" class="data row0 col0" >0.059000</td>
      <td id="T_c48d0_row0_col1" class="data row0 col1" >0.400000</td>
    </tr>
    <tr>
      <th id="T_c48d0_level0_row1" class="row_heading level0 row1" >287.6m < x <= 467.4m</th>
      <td id="T_c48d0_row1_col0" class="data row1 col0" >0.061000</td>
      <td id="T_c48d0_row1_col1" class="data row1 col1" >0.200000</td>
    </tr>
    <tr>
      <th id="T_c48d0_level0_row2" class="row_heading level0 row2" >467.4m < x <= 648.0m</th>
      <td id="T_c48d0_row2_col0" class="data row2 col0" >0.088000</td>
      <td id="T_c48d0_row2_col1" class="data row2 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_c48d0_level0_row3" class="row_heading level0 row3" >648.0m < x <= 3.8</th>
      <td id="T_c48d0_row3_col0" class="data row3 col0" >0.114000</td>
      <td id="T_c48d0_row3_col1" class="data row3 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_c48d0_level0_row4" class="row_heading level0 row4" >3.8 < x</th>
      <td id="T_c48d0_row4_col0" class="data row4 col0" >0.056000</td>
      <td id="T_c48d0_row4_col1" class="data row4 col1" >0.200000</td>
    </tr>
  </tbody>
</table>
    <style type="text/css">
#T_341ce_row0_col0 {
  background-color: #536edd;
  color: #f1f1f1;
}
#T_341ce_row0_col1, #T_341ce_row3_col0 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_341ce_row1_col0 {
  background-color: #6384eb;
  color: #f1f1f1;
}
#T_341ce_row1_col1 {
  background-color: #a9c6fd;
  color: #000000;
}
#T_341ce_row2_col0 {
  background-color: #b9d0f9;
  color: #000000;
}
#T_341ce_row2_col1, #T_341ce_row3_col1, #T_341ce_row4_col0 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_341ce_row4_col1 {
  background-color: #adc9fd;
  color: #000000;
}
</style>
<table id="T_341ce" style='display:inline'>
  <caption>X_dev distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_341ce_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_341ce_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_341ce_level0_row0" class="row_heading level0 row0" >x <= 31.1m</th>
      <td id="T_341ce_row0_col0" class="data row0 col0" >0.059000</td>
      <td id="T_341ce_row0_col1" class="data row0 col1" >0.401000</td>
    </tr>
    <tr>
      <th id="T_341ce_level0_row1" class="row_heading level0 row1" >287.6m < x <= 467.4m</th>
      <td id="T_341ce_row1_col0" class="data row1 col0" >0.062000</td>
      <td id="T_341ce_row1_col1" class="data row1 col1" >0.199000</td>
    </tr>
    <tr>
      <th id="T_341ce_level0_row2" class="row_heading level0 row2" >467.4m < x <= 648.0m</th>
      <td id="T_341ce_row2_col0" class="data row2 col0" >0.077000</td>
      <td id="T_341ce_row2_col1" class="data row2 col1" >0.099000</td>
    </tr>
    <tr>
      <th id="T_341ce_level0_row3" class="row_heading level0 row3" >648.0m < x <= 3.8</th>
      <td id="T_341ce_row3_col0" class="data row3 col0" >0.115000</td>
      <td id="T_341ce_row3_col1" class="data row3 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_341ce_level0_row4" class="row_heading level0 row4" >3.8 < x</th>
      <td id="T_341ce_row4_col0" class="data row4 col0" >0.054000</td>
      <td id="T_341ce_row4_col1" class="data row4 col1" >0.202000</td>
    </tr>
  </tbody>
</table>



    ------
    
    
    ------
    [AutoCarver] Fit RevolvingUtilizationOfUnsecuredLines (4/10)
    ---
    
     - [AutoCarver] Raw feature distribution
    


<style type="text/css">
#T_b6a5a_row0_col0, #T_b6a5a_row4_col0 {
  background-color: #4c66d6;
  color: #f1f1f1;
}
#T_b6a5a_row0_col1, #T_b6a5a_row1_col0, #T_b6a5a_row1_col1, #T_b6a5a_row2_col0, #T_b6a5a_row2_col1, #T_b6a5a_row3_col1, #T_b6a5a_row4_col1, #T_b6a5a_row5_col1, #T_b6a5a_row6_col1, #T_b6a5a_row7_col1 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_b6a5a_row3_col0 {
  background-color: #4257c9;
  color: #f1f1f1;
}
#T_b6a5a_row5_col0 {
  background-color: #6180e9;
  color: #f1f1f1;
}
#T_b6a5a_row6_col0 {
  background-color: #7ea1fa;
  color: #f1f1f1;
}
#T_b6a5a_row7_col0 {
  background-color: #c0d4f5;
  color: #000000;
}
#T_b6a5a_row8_col0, #T_b6a5a_row8_col1 {
  background-color: #b40426;
  color: #f1f1f1;
}
</style>
<table id="T_b6a5a" style='display:inline'>
  <caption>X distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_b6a5a_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_b6a5a_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
    <tr>
      <th class="index_name level0" >RevolvingUtilizationOfUnsecuredLines</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_b6a5a_level0_row0" class="row_heading level0 row0" >x <= 3.0m</th>
      <td id="T_b6a5a_row0_col0" class="data row0 col0" >0.025000</td>
      <td id="T_b6a5a_row0_col1" class="data row0 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_b6a5a_level0_row1" class="row_heading level0 row1" >3.0m < x <= 19.2m</th>
      <td id="T_b6a5a_row1_col0" class="data row1 col0" >0.014000</td>
      <td id="T_b6a5a_row1_col1" class="data row1 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_b6a5a_level0_row2" class="row_heading level0 row2" >19.2m < x <= 43.6m</th>
      <td id="T_b6a5a_row2_col0" class="data row2 col0" >0.014000</td>
      <td id="T_b6a5a_row2_col1" class="data row2 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_b6a5a_level0_row3" class="row_heading level0 row3" >43.6m < x <= 83.8m</th>
      <td id="T_b6a5a_row3_col0" class="data row3 col0" >0.019000</td>
      <td id="T_b6a5a_row3_col1" class="data row3 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_b6a5a_level0_row4" class="row_heading level0 row4" >83.8m < x <= 155.4m</th>
      <td id="T_b6a5a_row4_col0" class="data row4 col0" >0.025000</td>
      <td id="T_b6a5a_row4_col1" class="data row4 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_b6a5a_level0_row5" class="row_heading level0 row5" >155.4m < x <= 273.6m</th>
      <td id="T_b6a5a_row5_col0" class="data row5 col0" >0.037000</td>
      <td id="T_b6a5a_row5_col1" class="data row5 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_b6a5a_level0_row6" class="row_heading level0 row6" >273.6m < x <= 447.4m</th>
      <td id="T_b6a5a_row6_col0" class="data row6 col0" >0.053000</td>
      <td id="T_b6a5a_row6_col1" class="data row6 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_b6a5a_level0_row7" class="row_heading level0 row7" >447.4m < x <= 700.7m</th>
      <td id="T_b6a5a_row7_col0" class="data row7 col0" >0.088000</td>
      <td id="T_b6a5a_row7_col1" class="data row7 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_b6a5a_level0_row8" class="row_heading level0 row8" >700.7m < x</th>
      <td id="T_b6a5a_row8_col0" class="data row8 col0" >0.199000</td>
      <td id="T_b6a5a_row8_col1" class="data row8 col1" >0.200000</td>
    </tr>
  </tbody>
</table>
    <style type="text/css">
#T_af954_row0_col0 {
  background-color: #4f69d9;
  color: #f1f1f1;
}
#T_af954_row0_col1, #T_af954_row4_col1, #T_af954_row5_col1 {
  background-color: #4055c8;
  color: #f1f1f1;
}
#T_af954_row1_col0, #T_af954_row7_col1 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_af954_row1_col1, #T_af954_row2_col0, #T_af954_row6_col1 {
  background-color: #3d50c3;
  color: #f1f1f1;
}
#T_af954_row2_col1, #T_af954_row3_col1 {
  background-color: #4961d2;
  color: #f1f1f1;
}
#T_af954_row3_col0 {
  background-color: #455cce;
  color: #f1f1f1;
}
#T_af954_row4_col0 {
  background-color: #4a63d3;
  color: #f1f1f1;
}
#T_af954_row5_col0 {
  background-color: #5a78e4;
  color: #f1f1f1;
}
#T_af954_row6_col0 {
  background-color: #80a3fa;
  color: #f1f1f1;
}
#T_af954_row7_col0 {
  background-color: #c5d6f2;
  color: #000000;
}
#T_af954_row8_col0, #T_af954_row8_col1 {
  background-color: #b40426;
  color: #f1f1f1;
}
</style>
<table id="T_af954" style='display:inline'>
  <caption>X_dev distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_af954_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_af954_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
    <tr>
      <th class="index_name level0" >RevolvingUtilizationOfUnsecuredLines</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_af954_level0_row0" class="row_heading level0 row0" >x <= 3.0m</th>
      <td id="T_af954_row0_col0" class="data row0 col0" >0.025000</td>
      <td id="T_af954_row0_col1" class="data row0 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_af954_level0_row1" class="row_heading level0 row1" >3.0m < x <= 19.2m</th>
      <td id="T_af954_row1_col0" class="data row1 col0" >0.012000</td>
      <td id="T_af954_row1_col1" class="data row1 col1" >0.099000</td>
    </tr>
    <tr>
      <th id="T_af954_level0_row2" class="row_heading level0 row2" >19.2m < x <= 43.6m</th>
      <td id="T_af954_row2_col0" class="data row2 col0" >0.014000</td>
      <td id="T_af954_row2_col1" class="data row2 col1" >0.103000</td>
    </tr>
    <tr>
      <th id="T_af954_level0_row3" class="row_heading level0 row3" >43.6m < x <= 83.8m</th>
      <td id="T_af954_row3_col0" class="data row3 col0" >0.019000</td>
      <td id="T_af954_row3_col1" class="data row3 col1" >0.103000</td>
    </tr>
    <tr>
      <th id="T_af954_level0_row4" class="row_heading level0 row4" >83.8m < x <= 155.4m</th>
      <td id="T_af954_row4_col0" class="data row4 col0" >0.022000</td>
      <td id="T_af954_row4_col1" class="data row4 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_af954_level0_row5" class="row_heading level0 row5" >155.4m < x <= 273.6m</th>
      <td id="T_af954_row5_col0" class="data row5 col0" >0.031000</td>
      <td id="T_af954_row5_col1" class="data row5 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_af954_level0_row6" class="row_heading level0 row6" >273.6m < x <= 447.4m</th>
      <td id="T_af954_row6_col0" class="data row6 col0" >0.052000</td>
      <td id="T_af954_row6_col1" class="data row6 col1" >0.099000</td>
    </tr>
    <tr>
      <th id="T_af954_level0_row7" class="row_heading level0 row7" >447.4m < x <= 700.7m</th>
      <td id="T_af954_row7_col0" class="data row7 col0" >0.090000</td>
      <td id="T_af954_row7_col1" class="data row7 col1" >0.098000</td>
    </tr>
    <tr>
      <th id="T_af954_level0_row8" class="row_heading level0 row8" >700.7m < x</th>
      <td id="T_af954_row8_col0" class="data row8 col0" >0.199000</td>
      <td id="T_af954_row8_col1" class="data row8 col1" >0.198000</td>
    </tr>
  </tbody>
</table>



    Grouping modalities   : 100%|██████████| 162/162 [00:00<00:00, 6461.12it/s]
    Computing associations: 100%|██████████| 162/162 [00:00<00:00, 3819.84it/s]
    Testing robustness    :   0%|          | 0/162 [00:00<?, ?it/s]

    
     - [AutoCarver] Carved feature distribution
    

    
    


<style type="text/css">
#T_e726c_row0_col0, #T_e726c_row1_col1, #T_e726c_row2_col1, #T_e726c_row3_col1 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_e726c_row0_col1, #T_e726c_row4_col0 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_e726c_row1_col0 {
  background-color: #5875e1;
  color: #f1f1f1;
}
#T_e726c_row2_col0 {
  background-color: #7699f6;
  color: #f1f1f1;
}
#T_e726c_row3_col0 {
  background-color: #bad0f8;
  color: #000000;
}
#T_e726c_row4_col1 {
  background-color: #8db0fe;
  color: #000000;
}
</style>
<table id="T_e726c" style='display:inline'>
  <caption>X distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_e726c_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_e726c_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_e726c_level0_row0" class="row_heading level0 row0" >x <= 3.0m</th>
      <td id="T_e726c_row0_col0" class="data row0 col0" >0.020000</td>
      <td id="T_e726c_row0_col1" class="data row0 col1" >0.500000</td>
    </tr>
    <tr>
      <th id="T_e726c_level0_row1" class="row_heading level0 row1" >155.4m < x <= 273.6m</th>
      <td id="T_e726c_row1_col0" class="data row1 col0" >0.037000</td>
      <td id="T_e726c_row1_col1" class="data row1 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_e726c_level0_row2" class="row_heading level0 row2" >273.6m < x <= 447.4m</th>
      <td id="T_e726c_row2_col0" class="data row2 col0" >0.053000</td>
      <td id="T_e726c_row2_col1" class="data row2 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_e726c_level0_row3" class="row_heading level0 row3" >447.4m < x <= 700.7m</th>
      <td id="T_e726c_row3_col0" class="data row3 col0" >0.088000</td>
      <td id="T_e726c_row3_col1" class="data row3 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_e726c_level0_row4" class="row_heading level0 row4" >700.7m < x</th>
      <td id="T_e726c_row4_col0" class="data row4 col0" >0.199000</td>
      <td id="T_e726c_row4_col1" class="data row4 col1" >0.200000</td>
    </tr>
  </tbody>
</table>
    <style type="text/css">
#T_1612b_row0_col0, #T_1612b_row2_col1, #T_1612b_row3_col1 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_1612b_row0_col1, #T_1612b_row4_col0 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_1612b_row1_col0 {
  background-color: #506bda;
  color: #f1f1f1;
}
#T_1612b_row1_col1 {
  background-color: #3c4ec2;
  color: #f1f1f1;
}
#T_1612b_row2_col0 {
  background-color: #779af7;
  color: #f1f1f1;
}
#T_1612b_row3_col0 {
  background-color: #bfd3f6;
  color: #000000;
}
#T_1612b_row4_col1 {
  background-color: #8caffe;
  color: #000000;
}
</style>
<table id="T_1612b" style='display:inline'>
  <caption>X_dev distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_1612b_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_1612b_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_1612b_level0_row0" class="row_heading level0 row0" >x <= 3.0m</th>
      <td id="T_1612b_row0_col0" class="data row0 col0" >0.018000</td>
      <td id="T_1612b_row0_col1" class="data row0 col1" >0.504000</td>
    </tr>
    <tr>
      <th id="T_1612b_level0_row1" class="row_heading level0 row1" >155.4m < x <= 273.6m</th>
      <td id="T_1612b_row1_col0" class="data row1 col0" >0.031000</td>
      <td id="T_1612b_row1_col1" class="data row1 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_1612b_level0_row2" class="row_heading level0 row2" >273.6m < x <= 447.4m</th>
      <td id="T_1612b_row2_col0" class="data row2 col0" >0.052000</td>
      <td id="T_1612b_row2_col1" class="data row2 col1" >0.099000</td>
    </tr>
    <tr>
      <th id="T_1612b_level0_row3" class="row_heading level0 row3" >447.4m < x <= 700.7m</th>
      <td id="T_1612b_row3_col0" class="data row3 col0" >0.090000</td>
      <td id="T_1612b_row3_col1" class="data row3 col1" >0.098000</td>
    </tr>
    <tr>
      <th id="T_1612b_level0_row4" class="row_heading level0 row4" >700.7m < x</th>
      <td id="T_1612b_row4_col0" class="data row4 col0" >0.199000</td>
      <td id="T_1612b_row4_col1" class="data row4 col1" >0.198000</td>
    </tr>
  </tbody>
</table>



    ------
    
    
    ------
    [AutoCarver] Fit NumberOfTimes90DaysLate (5/10)
    ---
    
     - [AutoCarver] Raw feature distribution
    


<style type="text/css">
#T_95852_row0_col0, #T_95852_row0_col1 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
</style>
<table id="T_95852" style='display:inline'>
  <caption>X distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_95852_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_95852_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
    <tr>
      <th class="index_name level0" >NumberOfTimes90DaysLate</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_95852_level0_row0" class="row_heading level0 row0" >x <= nan</th>
      <td id="T_95852_row0_col0" class="data row0 col0" >0.067000</td>
      <td id="T_95852_row0_col1" class="data row0 col1" >1.000000</td>
    </tr>
  </tbody>
</table>
    <style type="text/css">
#T_32cc5_row0_col0, #T_32cc5_row0_col1 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
</style>
<table id="T_32cc5" style='display:inline'>
  <caption>X_dev distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_32cc5_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_32cc5_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
    <tr>
      <th class="index_name level0" >NumberOfTimes90DaysLate</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_32cc5_level0_row0" class="row_heading level0 row0" >x <= nan</th>
      <td id="T_32cc5_row0_col0" class="data row0 col0" >0.066000</td>
      <td id="T_32cc5_row0_col1" class="data row0 col1" >1.000000</td>
    </tr>
  </tbody>
</table>



     - [AutoCarver] No robust combination for feature 'NumberOfTimes90DaysLate' could be found. It will be ignored. You might have to increase the size of your test sample (test sample not representative of test sample for this feature) or you should consider dropping this features.
    ------
    
    
    ------
    [AutoCarver] Fit NumberOfTime60-89DaysPastDueNotWorse (6/10)
    ---
    
     - [AutoCarver] Raw feature distribution
    


<style type="text/css">
#T_c77b3_row0_col0, #T_c77b3_row0_col1 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
</style>
<table id="T_c77b3" style='display:inline'>
  <caption>X distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_c77b3_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_c77b3_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
    <tr>
      <th class="index_name level0" >NumberOfTime60-89DaysPastDueNotWorse</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_c77b3_level0_row0" class="row_heading level0 row0" >x <= nan</th>
      <td id="T_c77b3_row0_col0" class="data row0 col0" >0.067000</td>
      <td id="T_c77b3_row0_col1" class="data row0 col1" >1.000000</td>
    </tr>
  </tbody>
</table>
    <style type="text/css">
#T_8f1c2_row0_col0, #T_8f1c2_row0_col1 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
</style>
<table id="T_8f1c2" style='display:inline'>
  <caption>X_dev distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_8f1c2_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_8f1c2_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
    <tr>
      <th class="index_name level0" >NumberOfTime60-89DaysPastDueNotWorse</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_8f1c2_level0_row0" class="row_heading level0 row0" >x <= nan</th>
      <td id="T_8f1c2_row0_col0" class="data row0 col0" >0.066000</td>
      <td id="T_8f1c2_row0_col1" class="data row0 col1" >1.000000</td>
    </tr>
  </tbody>
</table>



     - [AutoCarver] No robust combination for feature 'NumberOfTime60-89DaysPastDueNotWorse' could be found. It will be ignored. You might have to increase the size of your test sample (test sample not representative of test sample for this feature) or you should consider dropping this features.
    ------
    
    
    ------
    [AutoCarver] Fit NumberOfOpenCreditLinesAndLoans (7/10)
    ---
    
     - [AutoCarver] Raw feature distribution
    


<style type="text/css">
#T_ba364_row0_col0, #T_ba364_row2_col1 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_ba364_row0_col1 {
  background-color: #97b8ff;
  color: #000000;
}
#T_ba364_row1_col0 {
  background-color: #7da0f9;
  color: #f1f1f1;
}
#T_ba364_row1_col1 {
  background-color: #bad0f8;
  color: #000000;
}
#T_ba364_row2_col0, #T_ba364_row4_col1 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_ba364_row3_col0, #T_ba364_row4_col0 {
  background-color: #688aef;
  color: #f1f1f1;
}
#T_ba364_row3_col1 {
  background-color: #8badfd;
  color: #000000;
}
#T_ba364_row5_col0 {
  background-color: #90b2fe;
  color: #000000;
}
#T_ba364_row5_col1 {
  background-color: #e1dad6;
  color: #000000;
}
</style>
<table id="T_ba364" style='display:inline'>
  <caption>X distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_ba364_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_ba364_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
    <tr>
      <th class="index_name level0" >NumberOfOpenCreditLinesAndLoans</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_ba364_level0_row0" class="row_heading level0 row0" >x <= 3.0</th>
      <td id="T_ba364_row0_col0" class="data row0 col0" >0.107000</td>
      <td id="T_ba364_row0_col1" class="data row0 col1" >0.147000</td>
    </tr>
    <tr>
      <th id="T_ba364_level0_row1" class="row_heading level0 row1" >3.0 < x <= 5.0</th>
      <td id="T_ba364_row1_col0" class="data row1 col0" >0.064000</td>
      <td id="T_ba364_row1_col1" class="data row1 col1" >0.163000</td>
    </tr>
    <tr>
      <th id="T_ba364_level0_row2" class="row_heading level0 row2" >5.0 < x <= 8.0</th>
      <td id="T_ba364_row2_col0" class="data row2 col0" >0.053000</td>
      <td id="T_ba364_row2_col1" class="data row2 col1" >0.262000</td>
    </tr>
    <tr>
      <th id="T_ba364_level0_row3" class="row_heading level0 row3" >8.0 < x <= 10.0</th>
      <td id="T_ba364_row3_col0" class="data row3 col0" >0.061000</td>
      <td id="T_ba364_row3_col1" class="data row3 col1" >0.141000</td>
    </tr>
    <tr>
      <th id="T_ba364_level0_row4" class="row_heading level0 row4" >10.0 < x <= 12.0</th>
      <td id="T_ba364_row4_col0" class="data row4 col0" >0.061000</td>
      <td id="T_ba364_row4_col1" class="data row4 col1" >0.102000</td>
    </tr>
    <tr>
      <th id="T_ba364_level0_row5" class="row_heading level0 row5" >12.0 < x</th>
      <td id="T_ba364_row5_col0" class="data row5 col0" >0.067000</td>
      <td id="T_ba364_row5_col1" class="data row5 col1" >0.185000</td>
    </tr>
  </tbody>
</table>
    <style type="text/css">
#T_42fbe_row0_col0, #T_42fbe_row2_col1 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_42fbe_row0_col1 {
  background-color: #96b7ff;
  color: #000000;
}
#T_42fbe_row1_col0 {
  background-color: #7699f6;
  color: #f1f1f1;
}
#T_42fbe_row1_col1 {
  background-color: #bad0f8;
  color: #000000;
}
#T_42fbe_row2_col0, #T_42fbe_row4_col1 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_42fbe_row3_col0 {
  background-color: #455cce;
  color: #f1f1f1;
}
#T_42fbe_row3_col1 {
  background-color: #82a6fb;
  color: #f1f1f1;
}
#T_42fbe_row4_col0 {
  background-color: #506bda;
  color: #f1f1f1;
}
#T_42fbe_row5_col0 {
  background-color: #89acfd;
  color: #000000;
}
#T_42fbe_row5_col1 {
  background-color: #dcdddd;
  color: #000000;
}
</style>
<table id="T_42fbe" style='display:inline'>
  <caption>X_dev distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_42fbe_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_42fbe_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
    <tr>
      <th class="index_name level0" >NumberOfOpenCreditLinesAndLoans</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_42fbe_level0_row0" class="row_heading level0 row0" >x <= 3.0</th>
      <td id="T_42fbe_row0_col0" class="data row0 col0" >0.107000</td>
      <td id="T_42fbe_row0_col1" class="data row0 col1" >0.147000</td>
    </tr>
    <tr>
      <th id="T_42fbe_level0_row1" class="row_heading level0 row1" >3.0 < x <= 5.0</th>
      <td id="T_42fbe_row1_col0" class="data row1 col0" >0.063000</td>
      <td id="T_42fbe_row1_col1" class="data row1 col1" >0.164000</td>
    </tr>
    <tr>
      <th id="T_42fbe_level0_row2" class="row_heading level0 row2" >5.0 < x <= 8.0</th>
      <td id="T_42fbe_row2_col0" class="data row2 col0" >0.053000</td>
      <td id="T_42fbe_row2_col1" class="data row2 col1" >0.265000</td>
    </tr>
    <tr>
      <th id="T_42fbe_level0_row3" class="row_heading level0 row3" >8.0 < x <= 10.0</th>
      <td id="T_42fbe_row3_col0" class="data row3 col0" >0.055000</td>
      <td id="T_42fbe_row3_col1" class="data row3 col1" >0.138000</td>
    </tr>
    <tr>
      <th id="T_42fbe_level0_row4" class="row_heading level0 row4" >10.0 < x <= 12.0</th>
      <td id="T_42fbe_row4_col0" class="data row4 col0" >0.057000</td>
      <td id="T_42fbe_row4_col1" class="data row4 col1" >0.102000</td>
    </tr>
    <tr>
      <th id="T_42fbe_level0_row5" class="row_heading level0 row5" >12.0 < x</th>
      <td id="T_42fbe_row5_col0" class="data row5 col0" >0.066000</td>
      <td id="T_42fbe_row5_col1" class="data row5 col1" >0.183000</td>
    </tr>
  </tbody>
</table>



    Grouping modalities   : 100%|██████████| 30/30 [00:00<00:00, 3600.36it/s]
    Computing associations: 100%|██████████| 30/30 [00:00<00:00, 3562.24it/s]
    Testing robustness    :   0%|          | 0/30 [00:00<?, ?it/s]
    

    
     - [AutoCarver] Carved feature distribution
    


<style type="text/css">
#T_74cb2_row0_col0, #T_74cb2_row2_col1 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_74cb2_row0_col1, #T_74cb2_row2_col0 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_74cb2_row1_col0 {
  background-color: #7da0f9;
  color: #f1f1f1;
}
#T_74cb2_row1_col1 {
  background-color: #6687ed;
  color: #f1f1f1;
}
#T_74cb2_row3_col0 {
  background-color: #688aef;
  color: #f1f1f1;
}
#T_74cb2_row3_col1 {
  background-color: #e7745b;
  color: #f1f1f1;
}
#T_74cb2_row4_col0 {
  background-color: #90b2fe;
  color: #000000;
}
#T_74cb2_row4_col1 {
  background-color: #a9c6fd;
  color: #000000;
}
</style>
<table id="T_74cb2" style='display:inline'>
  <caption>X distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_74cb2_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_74cb2_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_74cb2_level0_row0" class="row_heading level0 row0" >x <= 3.0</th>
      <td id="T_74cb2_row0_col0" class="data row0 col0" >0.107000</td>
      <td id="T_74cb2_row0_col1" class="data row0 col1" >0.147000</td>
    </tr>
    <tr>
      <th id="T_74cb2_level0_row1" class="row_heading level0 row1" >3.0 < x <= 5.0</th>
      <td id="T_74cb2_row1_col0" class="data row1 col0" >0.064000</td>
      <td id="T_74cb2_row1_col1" class="data row1 col1" >0.163000</td>
    </tr>
    <tr>
      <th id="T_74cb2_level0_row2" class="row_heading level0 row2" >5.0 < x <= 8.0</th>
      <td id="T_74cb2_row2_col0" class="data row2 col0" >0.053000</td>
      <td id="T_74cb2_row2_col1" class="data row2 col1" >0.262000</td>
    </tr>
    <tr>
      <th id="T_74cb2_level0_row3" class="row_heading level0 row3" >8.0 < x <= 10.0</th>
      <td id="T_74cb2_row3_col0" class="data row3 col0" >0.061000</td>
      <td id="T_74cb2_row3_col1" class="data row3 col1" >0.243000</td>
    </tr>
    <tr>
      <th id="T_74cb2_level0_row4" class="row_heading level0 row4" >12.0 < x</th>
      <td id="T_74cb2_row4_col0" class="data row4 col0" >0.067000</td>
      <td id="T_74cb2_row4_col1" class="data row4 col1" >0.185000</td>
    </tr>
  </tbody>
</table>
    <style type="text/css">
#T_1f5ae_row0_col0, #T_1f5ae_row2_col1 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_1f5ae_row0_col1, #T_1f5ae_row2_col0 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_1f5ae_row1_col0 {
  background-color: #7699f6;
  color: #f1f1f1;
}
#T_1f5ae_row1_col1 {
  background-color: #6788ee;
  color: #f1f1f1;
}
#T_1f5ae_row3_col0 {
  background-color: #4b64d5;
  color: #f1f1f1;
}
#T_1f5ae_row3_col1 {
  background-color: #f08a6c;
  color: #f1f1f1;
}
#T_1f5ae_row4_col0 {
  background-color: #89acfd;
  color: #000000;
}
#T_1f5ae_row4_col1 {
  background-color: #a1c0ff;
  color: #000000;
}
</style>
<table id="T_1f5ae" style='display:inline'>
  <caption>X_dev distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_1f5ae_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_1f5ae_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_1f5ae_level0_row0" class="row_heading level0 row0" >x <= 3.0</th>
      <td id="T_1f5ae_row0_col0" class="data row0 col0" >0.107000</td>
      <td id="T_1f5ae_row0_col1" class="data row0 col1" >0.147000</td>
    </tr>
    <tr>
      <th id="T_1f5ae_level0_row1" class="row_heading level0 row1" >3.0 < x <= 5.0</th>
      <td id="T_1f5ae_row1_col0" class="data row1 col0" >0.063000</td>
      <td id="T_1f5ae_row1_col1" class="data row1 col1" >0.164000</td>
    </tr>
    <tr>
      <th id="T_1f5ae_level0_row2" class="row_heading level0 row2" >5.0 < x <= 8.0</th>
      <td id="T_1f5ae_row2_col0" class="data row2 col0" >0.053000</td>
      <td id="T_1f5ae_row2_col1" class="data row2 col1" >0.265000</td>
    </tr>
    <tr>
      <th id="T_1f5ae_level0_row3" class="row_heading level0 row3" >8.0 < x <= 10.0</th>
      <td id="T_1f5ae_row3_col0" class="data row3 col0" >0.056000</td>
      <td id="T_1f5ae_row3_col1" class="data row3 col1" >0.240000</td>
    </tr>
    <tr>
      <th id="T_1f5ae_level0_row4" class="row_heading level0 row4" >12.0 < x</th>
      <td id="T_1f5ae_row4_col0" class="data row4 col0" >0.066000</td>
      <td id="T_1f5ae_row4_col1" class="data row4 col1" >0.183000</td>
    </tr>
  </tbody>
</table>



    ------
    
    
    ------
    [AutoCarver] Fit NumberRealEstateLoansOrLines (8/10)
    ---
    
     - [AutoCarver] Raw feature distribution
    


<style type="text/css">
#T_6c2b7_row0_col0, #T_6c2b7_row0_col1 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_6c2b7_row1_col0, #T_6c2b7_row2_col1 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_6c2b7_row1_col1 {
  background-color: #f7a688;
  color: #000000;
}
#T_6c2b7_row2_col0 {
  background-color: #c0d4f5;
  color: #000000;
}
</style>
<table id="T_6c2b7" style='display:inline'>
  <caption>X distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_6c2b7_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_6c2b7_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
    <tr>
      <th class="index_name level0" >NumberRealEstateLoansOrLines</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_6c2b7_level0_row0" class="row_heading level0 row0" >x <= 0.0</th>
      <td id="T_6c2b7_row0_col0" class="data row0 col0" >0.083000</td>
      <td id="T_6c2b7_row0_col1" class="data row0 col1" >0.376000</td>
    </tr>
    <tr>
      <th id="T_6c2b7_level0_row1" class="row_heading level0 row1" >0.0 < x <= 1.0</th>
      <td id="T_6c2b7_row1_col0" class="data row1 col0" >0.053000</td>
      <td id="T_6c2b7_row1_col1" class="data row1 col1" >0.348000</td>
    </tr>
    <tr>
      <th id="T_6c2b7_level0_row2" class="row_heading level0 row2" >1.0 < x</th>
      <td id="T_6c2b7_row2_col0" class="data row2 col0" >0.065000</td>
      <td id="T_6c2b7_row2_col1" class="data row2 col1" >0.277000</td>
    </tr>
  </tbody>
</table>
    <style type="text/css">
#T_f83ea_row0_col0, #T_f83ea_row0_col1 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_f83ea_row1_col0, #T_f83ea_row2_col1 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_f83ea_row1_col1 {
  background-color: #f08b6e;
  color: #f1f1f1;
}
#T_f83ea_row2_col0 {
  background-color: #84a7fc;
  color: #f1f1f1;
}
</style>
<table id="T_f83ea" style='display:inline'>
  <caption>X_dev distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_f83ea_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_f83ea_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
    <tr>
      <th class="index_name level0" >NumberRealEstateLoansOrLines</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_f83ea_level0_row0" class="row_heading level0 row0" >x <= 0.0</th>
      <td id="T_f83ea_row0_col0" class="data row0 col0" >0.083000</td>
      <td id="T_f83ea_row0_col1" class="data row0 col1" >0.373000</td>
    </tr>
    <tr>
      <th id="T_f83ea_level0_row1" class="row_heading level0 row1" >0.0 < x <= 1.0</th>
      <td id="T_f83ea_row1_col0" class="data row1 col0" >0.052000</td>
      <td id="T_f83ea_row1_col1" class="data row1 col1" >0.352000</td>
    </tr>
    <tr>
      <th id="T_f83ea_level0_row2" class="row_heading level0 row2" >1.0 < x</th>
      <td id="T_f83ea_row2_col0" class="data row2 col0" >0.059000</td>
      <td id="T_f83ea_row2_col1" class="data row2 col1" >0.276000</td>
    </tr>
  </tbody>
</table>



    Grouping modalities   : 100%|██████████| 3/3 [00:00<00:00, 1506.03it/s]
    Computing associations: 100%|██████████| 3/3 [00:00<00:00, 3008.83it/s]
    Testing robustness    :   0%|          | 0/3 [00:00<?, ?it/s]

    
     - [AutoCarver] Carved feature distribution
    

    
    


<style type="text/css">
#T_8eec6_row0_col0, #T_8eec6_row0_col1 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_8eec6_row1_col0, #T_8eec6_row2_col1 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_8eec6_row1_col1 {
  background-color: #f7a688;
  color: #000000;
}
#T_8eec6_row2_col0 {
  background-color: #c0d4f5;
  color: #000000;
}
</style>
<table id="T_8eec6" style='display:inline'>
  <caption>X distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_8eec6_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_8eec6_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_8eec6_level0_row0" class="row_heading level0 row0" >x <= 0.0</th>
      <td id="T_8eec6_row0_col0" class="data row0 col0" >0.083000</td>
      <td id="T_8eec6_row0_col1" class="data row0 col1" >0.376000</td>
    </tr>
    <tr>
      <th id="T_8eec6_level0_row1" class="row_heading level0 row1" >0.0 < x <= 1.0</th>
      <td id="T_8eec6_row1_col0" class="data row1 col0" >0.053000</td>
      <td id="T_8eec6_row1_col1" class="data row1 col1" >0.348000</td>
    </tr>
    <tr>
      <th id="T_8eec6_level0_row2" class="row_heading level0 row2" >1.0 < x</th>
      <td id="T_8eec6_row2_col0" class="data row2 col0" >0.065000</td>
      <td id="T_8eec6_row2_col1" class="data row2 col1" >0.277000</td>
    </tr>
  </tbody>
</table>
    <style type="text/css">
#T_812a8_row0_col0, #T_812a8_row0_col1 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_812a8_row1_col0, #T_812a8_row2_col1 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_812a8_row1_col1 {
  background-color: #f08b6e;
  color: #f1f1f1;
}
#T_812a8_row2_col0 {
  background-color: #84a7fc;
  color: #f1f1f1;
}
</style>
<table id="T_812a8" style='display:inline'>
  <caption>X_dev distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_812a8_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_812a8_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_812a8_level0_row0" class="row_heading level0 row0" >x <= 0.0</th>
      <td id="T_812a8_row0_col0" class="data row0 col0" >0.083000</td>
      <td id="T_812a8_row0_col1" class="data row0 col1" >0.373000</td>
    </tr>
    <tr>
      <th id="T_812a8_level0_row1" class="row_heading level0 row1" >0.0 < x <= 1.0</th>
      <td id="T_812a8_row1_col0" class="data row1 col0" >0.052000</td>
      <td id="T_812a8_row1_col1" class="data row1 col1" >0.352000</td>
    </tr>
    <tr>
      <th id="T_812a8_level0_row2" class="row_heading level0 row2" >1.0 < x</th>
      <td id="T_812a8_row2_col0" class="data row2 col0" >0.059000</td>
      <td id="T_812a8_row2_col1" class="data row2 col1" >0.276000</td>
    </tr>
  </tbody>
</table>



    ------
    
    
    ------
    [AutoCarver] Fit age (9/10)
    ---
    
     - [AutoCarver] Raw feature distribution
    


<style type="text/css">
#T_be9f2_row0_col0, #T_be9f2_row5_col1 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_be9f2_row0_col1 {
  background-color: #536edd;
  color: #f1f1f1;
}
#T_be9f2_row1_col0 {
  background-color: #ee8669;
  color: #f1f1f1;
}
#T_be9f2_row1_col1, #T_be9f2_row5_col0 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_be9f2_row2_col0 {
  background-color: #f7bca1;
  color: #000000;
}
#T_be9f2_row2_col1 {
  background-color: #efcfbf;
  color: #000000;
}
#T_be9f2_row3_col0 {
  background-color: #dbdcde;
  color: #000000;
}
#T_be9f2_row3_col1 {
  background-color: #e3d9d3;
  color: #000000;
}
#T_be9f2_row4_col0 {
  background-color: #92b4fe;
  color: #000000;
}
#T_be9f2_row4_col1 {
  background-color: #4f69d9;
  color: #f1f1f1;
}
</style>
<table id="T_be9f2" style='display:inline'>
  <caption>X distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_be9f2_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_be9f2_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
    <tr>
      <th class="index_name level0" >age</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_be9f2_level0_row0" class="row_heading level0 row0" >x <= 33.0</th>
      <td id="T_be9f2_row0_col0" class="data row0 col0" >0.115000</td>
      <td id="T_be9f2_row0_col1" class="data row0 col1" >0.114000</td>
    </tr>
    <tr>
      <th id="T_be9f2_level0_row1" class="row_heading level0 row1" >33.0 < x <= 39.0</th>
      <td id="T_be9f2_row1_col0" class="data row1 col0" >0.097000</td>
      <td id="T_be9f2_row1_col1" class="data row1 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_be9f2_level0_row2" class="row_heading level0 row2" >39.0 < x <= 48.0</th>
      <td id="T_be9f2_row2_col0" class="data row2 col0" >0.085000</td>
      <td id="T_be9f2_row2_col1" class="data row2 col1" >0.203000</td>
    </tr>
    <tr>
      <th id="T_be9f2_level0_row3" class="row_heading level0 row3" >48.0 < x <= 56.0</th>
      <td id="T_be9f2_row3_col0" class="data row3 col0" >0.071000</td>
      <td id="T_be9f2_row3_col1" class="data row3 col1" >0.193000</td>
    </tr>
    <tr>
      <th id="T_be9f2_level0_row4" class="row_heading level0 row4" >56.0 < x <= 61.0</th>
      <td id="T_be9f2_row4_col0" class="data row4 col0" >0.051000</td>
      <td id="T_be9f2_row4_col1" class="data row4 col1" >0.112000</td>
    </tr>
    <tr>
      <th id="T_be9f2_level0_row5" class="row_heading level0 row5" >61.0 < x</th>
      <td id="T_be9f2_row5_col0" class="data row5 col0" >0.028000</td>
      <td id="T_be9f2_row5_col1" class="data row5 col1" >0.277000</td>
    </tr>
  </tbody>
</table>
    <style type="text/css">
#T_59ed5_row0_col0, #T_59ed5_row5_col1 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_59ed5_row0_col1 {
  background-color: #5572df;
  color: #f1f1f1;
}
#T_59ed5_row1_col0 {
  background-color: #ed8366;
  color: #f1f1f1;
}
#T_59ed5_row1_col1, #T_59ed5_row5_col0 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_59ed5_row2_col0 {
  background-color: #f7bca1;
  color: #000000;
}
#T_59ed5_row2_col1 {
  background-color: #f1cdba;
  color: #000000;
}
#T_59ed5_row3_col0 {
  background-color: #e4d9d2;
  color: #000000;
}
#T_59ed5_row3_col1 {
  background-color: #e6d7cf;
  color: #000000;
}
#T_59ed5_row4_col0 {
  background-color: #88abfd;
  color: #000000;
}
#T_59ed5_row4_col1 {
  background-color: #5470de;
  color: #f1f1f1;
}
</style>
<table id="T_59ed5" style='display:inline'>
  <caption>X_dev distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_59ed5_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_59ed5_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
    <tr>
      <th class="index_name level0" >age</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_59ed5_level0_row0" class="row_heading level0 row0" >x <= 33.0</th>
      <td id="T_59ed5_row0_col0" class="data row0 col0" >0.110000</td>
      <td id="T_59ed5_row0_col1" class="data row0 col1" >0.114000</td>
    </tr>
    <tr>
      <th id="T_59ed5_level0_row1" class="row_heading level0 row1" >33.0 < x <= 39.0</th>
      <td id="T_59ed5_row1_col0" class="data row1 col0" >0.094000</td>
      <td id="T_59ed5_row1_col1" class="data row1 col1" >0.098000</td>
    </tr>
    <tr>
      <th id="T_59ed5_level0_row2" class="row_heading level0 row2" >39.0 < x <= 48.0</th>
      <td id="T_59ed5_row2_col0" class="data row2 col0" >0.082000</td>
      <td id="T_59ed5_row2_col1" class="data row2 col1" >0.204000</td>
    </tr>
    <tr>
      <th id="T_59ed5_level0_row3" class="row_heading level0 row3" >48.0 < x <= 56.0</th>
      <td id="T_59ed5_row3_col0" class="data row3 col0" >0.072000</td>
      <td id="T_59ed5_row3_col1" class="data row3 col1" >0.194000</td>
    </tr>
    <tr>
      <th id="T_59ed5_level0_row4" class="row_heading level0 row4" >56.0 < x <= 61.0</th>
      <td id="T_59ed5_row4_col0" class="data row4 col0" >0.048000</td>
      <td id="T_59ed5_row4_col1" class="data row4 col1" >0.113000</td>
    </tr>
    <tr>
      <th id="T_59ed5_level0_row5" class="row_heading level0 row5" >61.0 < x</th>
      <td id="T_59ed5_row5_col0" class="data row5 col0" >0.029000</td>
      <td id="T_59ed5_row5_col1" class="data row5 col1" >0.277000</td>
    </tr>
  </tbody>
</table>



    Grouping modalities   : 100%|██████████| 30/30 [00:00<00:00, 3313.21it/s]
    Computing associations: 100%|██████████| 30/30 [00:00<00:00, 2360.64it/s]
    Testing robustness    :   0%|          | 0/30 [00:00<?, ?it/s]

    
     - [AutoCarver] Carved feature distribution
    

    
    


<style type="text/css">
#T_aa7fe_row0_col0, #T_aa7fe_row1_col1 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_aa7fe_row0_col1 {
  background-color: #3d50c3;
  color: #f1f1f1;
}
#T_aa7fe_row1_col0 {
  background-color: #f7ac8e;
  color: #000000;
}
#T_aa7fe_row2_col0 {
  background-color: #dbdcde;
  color: #000000;
}
#T_aa7fe_row2_col1 {
  background-color: #c7d7f0;
  color: #000000;
}
#T_aa7fe_row3_col0 {
  background-color: #92b4fe;
  color: #000000;
}
#T_aa7fe_row3_col1, #T_aa7fe_row4_col0 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_aa7fe_row4_col1 {
  background-color: #e16751;
  color: #f1f1f1;
}
</style>
<table id="T_aa7fe" style='display:inline'>
  <caption>X distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_aa7fe_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_aa7fe_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_aa7fe_level0_row0" class="row_heading level0 row0" >x <= 33.0</th>
      <td id="T_aa7fe_row0_col0" class="data row0 col0" >0.115000</td>
      <td id="T_aa7fe_row0_col1" class="data row0 col1" >0.114000</td>
    </tr>
    <tr>
      <th id="T_aa7fe_level0_row1" class="row_heading level0 row1" >33.0 < x <= 39.0</th>
      <td id="T_aa7fe_row1_col0" class="data row1 col0" >0.089000</td>
      <td id="T_aa7fe_row1_col1" class="data row1 col1" >0.304000</td>
    </tr>
    <tr>
      <th id="T_aa7fe_level0_row2" class="row_heading level0 row2" >48.0 < x <= 56.0</th>
      <td id="T_aa7fe_row2_col0" class="data row2 col0" >0.071000</td>
      <td id="T_aa7fe_row2_col1" class="data row2 col1" >0.193000</td>
    </tr>
    <tr>
      <th id="T_aa7fe_level0_row3" class="row_heading level0 row3" >56.0 < x <= 61.0</th>
      <td id="T_aa7fe_row3_col0" class="data row3 col0" >0.051000</td>
      <td id="T_aa7fe_row3_col1" class="data row3 col1" >0.112000</td>
    </tr>
    <tr>
      <th id="T_aa7fe_level0_row4" class="row_heading level0 row4" >61.0 < x</th>
      <td id="T_aa7fe_row4_col0" class="data row4 col0" >0.028000</td>
      <td id="T_aa7fe_row4_col1" class="data row4 col1" >0.277000</td>
    </tr>
  </tbody>
</table>
    <style type="text/css">
#T_62490_row0_col0, #T_62490_row1_col1 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_62490_row0_col1 {
  background-color: #3c4ec2;
  color: #f1f1f1;
}
#T_62490_row1_col0 {
  background-color: #f7aa8c;
  color: #000000;
}
#T_62490_row2_col0 {
  background-color: #e4d9d2;
  color: #000000;
}
#T_62490_row2_col1 {
  background-color: #c9d7f0;
  color: #000000;
}
#T_62490_row3_col0 {
  background-color: #88abfd;
  color: #000000;
}
#T_62490_row3_col1, #T_62490_row4_col0 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_62490_row4_col1 {
  background-color: #df634e;
  color: #f1f1f1;
}
</style>
<table id="T_62490" style='display:inline'>
  <caption>X_dev distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_62490_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_62490_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_62490_level0_row0" class="row_heading level0 row0" >x <= 33.0</th>
      <td id="T_62490_row0_col0" class="data row0 col0" >0.110000</td>
      <td id="T_62490_row0_col1" class="data row0 col1" >0.114000</td>
    </tr>
    <tr>
      <th id="T_62490_level0_row1" class="row_heading level0 row1" >33.0 < x <= 39.0</th>
      <td id="T_62490_row1_col0" class="data row1 col0" >0.086000</td>
      <td id="T_62490_row1_col1" class="data row1 col1" >0.302000</td>
    </tr>
    <tr>
      <th id="T_62490_level0_row2" class="row_heading level0 row2" >48.0 < x <= 56.0</th>
      <td id="T_62490_row2_col0" class="data row2 col0" >0.072000</td>
      <td id="T_62490_row2_col1" class="data row2 col1" >0.194000</td>
    </tr>
    <tr>
      <th id="T_62490_level0_row3" class="row_heading level0 row3" >56.0 < x <= 61.0</th>
      <td id="T_62490_row3_col0" class="data row3 col0" >0.048000</td>
      <td id="T_62490_row3_col1" class="data row3 col1" >0.113000</td>
    </tr>
    <tr>
      <th id="T_62490_level0_row4" class="row_heading level0 row4" >61.0 < x</th>
      <td id="T_62490_row4_col0" class="data row4 col0" >0.029000</td>
      <td id="T_62490_row4_col1" class="data row4 col1" >0.277000</td>
    </tr>
  </tbody>
</table>



    ------
    
    
    ------
    [AutoCarver] Fit MonthlyIncome (10/10)
    ---
    
     - [AutoCarver] Raw feature distribution
    


<style type="text/css">
#T_9574c_row0_col0 {
  background-color: #f18f71;
  color: #f1f1f1;
}
#T_9574c_row0_col1, #T_9574c_row1_col1 {
  background-color: #3d50c3;
  color: #f1f1f1;
}
#T_9574c_row1_col0, #T_9574c_row2_col1 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_9574c_row2_col0 {
  background-color: #f7b497;
  color: #000000;
}
#T_9574c_row3_col0 {
  background-color: #a6c4fe;
  color: #000000;
}
#T_9574c_row3_col1 {
  background-color: #bb1b2c;
  color: #f1f1f1;
}
#T_9574c_row4_col0 {
  background-color: #6384eb;
  color: #f1f1f1;
}
#T_9574c_row4_col1, #T_9574c_row5_col0, #T_9574c_row5_col1 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_9574c_row6_col0 {
  background-color: #84a7fc;
  color: #f1f1f1;
}
#T_9574c_row6_col1 {
  background-color: #c32e31;
  color: #f1f1f1;
}
</style>
<table id="T_9574c" style='display:inline'>
  <caption>X distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_9574c_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_9574c_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
    <tr>
      <th class="index_name level0" >MonthlyIncome</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_9574c_level0_row0" class="row_heading level0 row0" >x <= 2.3K</th>
      <td id="T_9574c_row0_col0" class="data row0 col0" >0.085000</td>
      <td id="T_9574c_row0_col1" class="data row0 col1" >0.101000</td>
    </tr>
    <tr>
      <th id="T_9574c_level0_row1" class="row_heading level0 row1" >2.3K < x <= 3.4K</th>
      <td id="T_9574c_row1_col0" class="data row1 col0" >0.097000</td>
      <td id="T_9574c_row1_col1" class="data row1 col1" >0.101000</td>
    </tr>
    <tr>
      <th id="T_9574c_level0_row2" class="row_heading level0 row2" >3.4K < x <= 5.4K</th>
      <td id="T_9574c_row2_col0" class="data row2 col0" >0.080000</td>
      <td id="T_9574c_row2_col1" class="data row2 col1" >0.201000</td>
    </tr>
    <tr>
      <th id="T_9574c_level0_row3" class="row_heading level0 row3" >5.4K < x <= 8.2K</th>
      <td id="T_9574c_row3_col0" class="data row3 col0" >0.061000</td>
      <td id="T_9574c_row3_col1" class="data row3 col1" >0.199000</td>
    </tr>
    <tr>
      <th id="T_9574c_level0_row4" class="row_heading level0 row4" >8.2K < x <= 10.7K</th>
      <td id="T_9574c_row4_col0" class="data row4 col0" >0.051000</td>
      <td id="T_9574c_row4_col1" class="data row4 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_9574c_level0_row5" class="row_heading level0 row5" >10.7K < x</th>
      <td id="T_9574c_row5_col0" class="data row5 col0" >0.044000</td>
      <td id="T_9574c_row5_col1" class="data row5 col1" >0.100000</td>
    </tr>
    <tr>
      <th id="T_9574c_level0_row6" class="row_heading level0 row6" >__NAN__</th>
      <td id="T_9574c_row6_col0" class="data row6 col0" >0.056000</td>
      <td id="T_9574c_row6_col1" class="data row6 col1" >0.197000</td>
    </tr>
  </tbody>
</table>
    <style type="text/css">
#T_95e47_row0_col0 {
  background-color: #e67259;
  color: #f1f1f1;
}
#T_95e47_row0_col1, #T_95e47_row1_col1 {
  background-color: #4358cb;
  color: #f1f1f1;
}
#T_95e47_row1_col0, #T_95e47_row6_col1 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_95e47_row2_col0 {
  background-color: #f0cdbb;
  color: #000000;
}
#T_95e47_row2_col1, #T_95e47_row3_col1 {
  background-color: #b70d28;
  color: #f1f1f1;
}
#T_95e47_row3_col0 {
  background-color: #a6c4fe;
  color: #000000;
}
#T_95e47_row4_col0, #T_95e47_row4_col1 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_95e47_row5_col0 {
  background-color: #5572df;
  color: #f1f1f1;
}
#T_95e47_row5_col1 {
  background-color: #465ecf;
  color: #f1f1f1;
}
#T_95e47_row6_col0 {
  background-color: #8caffe;
  color: #000000;
}
</style>
<table id="T_95e47" style='display:inline'>
  <caption>X_dev distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_95e47_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_95e47_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
    <tr>
      <th class="index_name level0" >MonthlyIncome</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_95e47_level0_row0" class="row_heading level0 row0" >x <= 2.3K</th>
      <td id="T_95e47_row0_col0" class="data row0 col0" >0.089000</td>
      <td id="T_95e47_row0_col1" class="data row0 col1" >0.101000</td>
    </tr>
    <tr>
      <th id="T_95e47_level0_row1" class="row_heading level0 row1" >2.3K < x <= 3.4K</th>
      <td id="T_95e47_row1_col0" class="data row1 col0" >0.098000</td>
      <td id="T_95e47_row1_col1" class="data row1 col1" >0.101000</td>
    </tr>
    <tr>
      <th id="T_95e47_level0_row2" class="row_heading level0 row2" >3.4K < x <= 5.4K</th>
      <td id="T_95e47_row2_col0" class="data row2 col0" >0.075000</td>
      <td id="T_95e47_row2_col1" class="data row2 col1" >0.199000</td>
    </tr>
    <tr>
      <th id="T_95e47_level0_row3" class="row_heading level0 row3" >5.4K < x <= 8.2K</th>
      <td id="T_95e47_row3_col0" class="data row3 col0" >0.060000</td>
      <td id="T_95e47_row3_col1" class="data row3 col1" >0.199000</td>
    </tr>
    <tr>
      <th id="T_95e47_level0_row4" class="row_heading level0 row4" >8.2K < x <= 10.7K</th>
      <td id="T_95e47_row4_col0" class="data row4 col0" >0.042000</td>
      <td id="T_95e47_row4_col1" class="data row4 col1" >0.098000</td>
    </tr>
    <tr>
      <th id="T_95e47_level0_row5" class="row_heading level0 row5" >10.7K < x</th>
      <td id="T_95e47_row5_col0" class="data row5 col0" >0.047000</td>
      <td id="T_95e47_row5_col1" class="data row5 col1" >0.102000</td>
    </tr>
    <tr>
      <th id="T_95e47_level0_row6" class="row_heading level0 row6" >__NAN__</th>
      <td id="T_95e47_row6_col0" class="data row6 col0" >0.056000</td>
      <td id="T_95e47_row6_col1" class="data row6 col1" >0.200000</td>
    </tr>
  </tbody>
</table>



    Grouping modalities   : 100%|██████████| 30/30 [00:00<00:00, 3022.63it/s]
    Computing associations: 100%|██████████| 30/30 [00:00<00:00, 2698.00it/s]
    Testing robustness    :   0%|          | 0/30 [00:00<?, ?it/s]

    
     - [AutoCarver] Carved feature distribution
    

    
    


<style type="text/css">
#T_ff8c8_row0_col0 {
  background-color: #f39577;
  color: #000000;
}
#T_ff8c8_row0_col1, #T_ff8c8_row1_col1, #T_ff8c8_row4_col0 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_ff8c8_row1_col0, #T_ff8c8_row2_col1, #T_ff8c8_row4_col1 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_ff8c8_row2_col0 {
  background-color: #f7ba9f;
  color: #000000;
}
#T_ff8c8_row3_col0 {
  background-color: #97b8ff;
  color: #000000;
}
#T_ff8c8_row3_col1 {
  background-color: #bb1b2c;
  color: #f1f1f1;
}
#T_ff8c8_row5_col0 {
  background-color: #7597f6;
  color: #f1f1f1;
}
#T_ff8c8_row5_col1 {
  background-color: #c32e31;
  color: #f1f1f1;
}
</style>
<table id="T_ff8c8" style='display:inline'>
  <caption>X distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_ff8c8_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_ff8c8_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_ff8c8_level0_row0" class="row_heading level0 row0" >x <= 2.3K</th>
      <td id="T_ff8c8_row0_col0" class="data row0 col0" >0.085000</td>
      <td id="T_ff8c8_row0_col1" class="data row0 col1" >0.101000</td>
    </tr>
    <tr>
      <th id="T_ff8c8_level0_row1" class="row_heading level0 row1" >2.3K < x <= 3.4K</th>
      <td id="T_ff8c8_row1_col0" class="data row1 col0" >0.097000</td>
      <td id="T_ff8c8_row1_col1" class="data row1 col1" >0.101000</td>
    </tr>
    <tr>
      <th id="T_ff8c8_level0_row2" class="row_heading level0 row2" >3.4K < x <= 5.4K</th>
      <td id="T_ff8c8_row2_col0" class="data row2 col0" >0.080000</td>
      <td id="T_ff8c8_row2_col1" class="data row2 col1" >0.201000</td>
    </tr>
    <tr>
      <th id="T_ff8c8_level0_row3" class="row_heading level0 row3" >5.4K < x <= 8.2K</th>
      <td id="T_ff8c8_row3_col0" class="data row3 col0" >0.061000</td>
      <td id="T_ff8c8_row3_col1" class="data row3 col1" >0.199000</td>
    </tr>
    <tr>
      <th id="T_ff8c8_level0_row4" class="row_heading level0 row4" >8.2K < x <= 10.7K</th>
      <td id="T_ff8c8_row4_col0" class="data row4 col0" >0.047000</td>
      <td id="T_ff8c8_row4_col1" class="data row4 col1" >0.201000</td>
    </tr>
    <tr>
      <th id="T_ff8c8_level0_row5" class="row_heading level0 row5" >__NAN__</th>
      <td id="T_ff8c8_row5_col0" class="data row5 col0" >0.056000</td>
      <td id="T_ff8c8_row5_col1" class="data row5 col1" >0.197000</td>
    </tr>
  </tbody>
</table>
    <style type="text/css">
#T_2460a_row0_col0 {
  background-color: #e7745b;
  color: #f1f1f1;
}
#T_2460a_row0_col1, #T_2460a_row1_col1, #T_2460a_row4_col0 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_2460a_row1_col0, #T_2460a_row4_col1, #T_2460a_row5_col1 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_2460a_row2_col0 {
  background-color: #edd1c2;
  color: #000000;
}
#T_2460a_row2_col1, #T_2460a_row3_col1 {
  background-color: #b70d28;
  color: #f1f1f1;
}
#T_2460a_row3_col0 {
  background-color: #9dbdff;
  color: #000000;
}
#T_2460a_row5_col0 {
  background-color: #82a6fb;
  color: #f1f1f1;
}
</style>
<table id="T_2460a" style='display:inline'>
  <caption>X_dev distribution</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_2460a_level0_col0" class="col_heading level0 col0" >target_rate</th>
      <th id="T_2460a_level0_col1" class="col_heading level0 col1" >frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_2460a_level0_row0" class="row_heading level0 row0" >x <= 2.3K</th>
      <td id="T_2460a_row0_col0" class="data row0 col0" >0.089000</td>
      <td id="T_2460a_row0_col1" class="data row0 col1" >0.101000</td>
    </tr>
    <tr>
      <th id="T_2460a_level0_row1" class="row_heading level0 row1" >2.3K < x <= 3.4K</th>
      <td id="T_2460a_row1_col0" class="data row1 col0" >0.098000</td>
      <td id="T_2460a_row1_col1" class="data row1 col1" >0.101000</td>
    </tr>
    <tr>
      <th id="T_2460a_level0_row2" class="row_heading level0 row2" >3.4K < x <= 5.4K</th>
      <td id="T_2460a_row2_col0" class="data row2 col0" >0.075000</td>
      <td id="T_2460a_row2_col1" class="data row2 col1" >0.199000</td>
    </tr>
    <tr>
      <th id="T_2460a_level0_row3" class="row_heading level0 row3" >5.4K < x <= 8.2K</th>
      <td id="T_2460a_row3_col0" class="data row3 col0" >0.060000</td>
      <td id="T_2460a_row3_col1" class="data row3 col1" >0.199000</td>
    </tr>
    <tr>
      <th id="T_2460a_level0_row4" class="row_heading level0 row4" >8.2K < x <= 10.7K</th>
      <td id="T_2460a_row4_col0" class="data row4 col0" >0.044000</td>
      <td id="T_2460a_row4_col1" class="data row4 col1" >0.200000</td>
    </tr>
    <tr>
      <th id="T_2460a_level0_row5" class="row_heading level0 row5" >__NAN__</th>
      <td id="T_2460a_row5_col0" class="data row5 col0" >0.056000</td>
      <td id="T_2460a_row5_col1" class="data row5 col1" >0.200000</td>
    </tr>
  </tbody>
</table>



    ------
    
     - [BaseDiscretizer] Transform Quantitative ['age', 'NumberOfDependents', 'DebtRatio', 'RevolvingUtilizationOfUnsecuredLines', 'NumberOfOpenCreditLinesAndLoans', 'NumberRealEstateLoansOrLines', 'NumberOfTime30-59DaysPastDueNotWorse', 'MonthlyIncome']
    

## Inspecting Discretization


```python
x_discretized[quantitative_features].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RevolvingUtilizationOfUnsecuredLines</th>
      <th>age</th>
      <th>NumberOfTime30-59DaysPastDueNotWorse</th>
      <th>DebtRatio</th>
      <th>MonthlyIncome</th>
      <th>NumberOfOpenCreditLinesAndLoans</th>
      <th>NumberOfTimes90DaysLate</th>
      <th>NumberRealEstateLoansOrLines</th>
      <th>NumberOfTime60-89DaysPastDueNotWorse</th>
      <th>NumberOfDependents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>87936</th>
      <td>0.0</td>
      <td>4</td>
      <td>0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3893</th>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>41405</th>
      <td>4.0</td>
      <td>1</td>
      <td>0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>91125</th>
      <td>2.0</td>
      <td>2</td>
      <td>0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>67373</th>
      <td>3.0</td>
      <td>2</td>
      <td>3</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
auto_carver.summary()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>label</th>
      <th>content</th>
    </tr>
    <tr>
      <th>feature</th>
      <th>dtype</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">DebtRatio</th>
      <th>float</th>
      <td>0</td>
      <td>[x &lt;= 287.6m]</td>
    </tr>
    <tr>
      <th>float</th>
      <td>1</td>
      <td>[287.6m &lt; x &lt;= 467.4m]</td>
    </tr>
    <tr>
      <th>float</th>
      <td>2</td>
      <td>[467.4m &lt; x &lt;= 648.0m]</td>
    </tr>
    <tr>
      <th>float</th>
      <td>3</td>
      <td>[648.0m &lt; x &lt;= 3.8]</td>
    </tr>
    <tr>
      <th>float</th>
      <td>4</td>
      <td>[3.8 &lt; x]</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">MonthlyIncome</th>
      <th>float</th>
      <td>0</td>
      <td>[x &lt;= 2.3K]</td>
    </tr>
    <tr>
      <th>float</th>
      <td>1</td>
      <td>[2.3K &lt; x &lt;= 3.4K]</td>
    </tr>
    <tr>
      <th>float</th>
      <td>2</td>
      <td>[3.4K &lt; x &lt;= 5.4K]</td>
    </tr>
    <tr>
      <th>float</th>
      <td>3</td>
      <td>[5.4K &lt; x &lt;= 8.2K]</td>
    </tr>
    <tr>
      <th>float</th>
      <td>4</td>
      <td>[8.2K &lt; x]</td>
    </tr>
    <tr>
      <th>float</th>
      <td>5</td>
      <td>[__NAN__]</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">NumberOfDependents</th>
      <th>float</th>
      <td>0</td>
      <td>[x &lt;= 0.0]</td>
    </tr>
    <tr>
      <th>float</th>
      <td>1</td>
      <td>[0.0 &lt; x &lt;= 1.0]</td>
    </tr>
    <tr>
      <th>float</th>
      <td>2</td>
      <td>[1.0 &lt; x]</td>
    </tr>
    <tr>
      <th>float</th>
      <td>3</td>
      <td>[__NAN__]</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">NumberOfOpenCreditLinesAndLoans</th>
      <th>float</th>
      <td>0</td>
      <td>[x &lt;= 3.0]</td>
    </tr>
    <tr>
      <th>float</th>
      <td>1</td>
      <td>[3.0 &lt; x &lt;= 5.0]</td>
    </tr>
    <tr>
      <th>float</th>
      <td>2</td>
      <td>[5.0 &lt; x &lt;= 8.0]</td>
    </tr>
    <tr>
      <th>float</th>
      <td>3</td>
      <td>[8.0 &lt; x &lt;= 12.0]</td>
    </tr>
    <tr>
      <th>float</th>
      <td>4</td>
      <td>[12.0 &lt; x]</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">NumberOfTime30-59DaysPastDueNotWorse</th>
      <th>float</th>
      <td>0</td>
      <td>[x &lt;= 0.0]</td>
    </tr>
    <tr>
      <th>float</th>
      <td>1</td>
      <td>[0.0 &lt; x]</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">NumberRealEstateLoansOrLines</th>
      <th>float</th>
      <td>0</td>
      <td>[x &lt;= 0.0]</td>
    </tr>
    <tr>
      <th>float</th>
      <td>1</td>
      <td>[0.0 &lt; x &lt;= 1.0]</td>
    </tr>
    <tr>
      <th>float</th>
      <td>2</td>
      <td>[1.0 &lt; x]</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">RevolvingUtilizationOfUnsecuredLines</th>
      <th>float</th>
      <td>0</td>
      <td>[x &lt;= 155.4m]</td>
    </tr>
    <tr>
      <th>float</th>
      <td>1</td>
      <td>[155.4m &lt; x &lt;= 273.6m]</td>
    </tr>
    <tr>
      <th>float</th>
      <td>2</td>
      <td>[273.6m &lt; x &lt;= 447.4m]</td>
    </tr>
    <tr>
      <th>float</th>
      <td>3</td>
      <td>[447.4m &lt; x &lt;= 700.7m]</td>
    </tr>
    <tr>
      <th>float</th>
      <td>4</td>
      <td>[700.7m &lt; x]</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">age</th>
      <th>float</th>
      <td>0</td>
      <td>[x &lt;= 33.0]</td>
    </tr>
    <tr>
      <th>float</th>
      <td>1</td>
      <td>[33.0 &lt; x &lt;= 48.0]</td>
    </tr>
    <tr>
      <th>float</th>
      <td>2</td>
      <td>[48.0 &lt; x &lt;= 56.0]</td>
    </tr>
    <tr>
      <th>float</th>
      <td>3</td>
      <td>[56.0 &lt; x &lt;= 61.0]</td>
    </tr>
    <tr>
      <th>float</th>
      <td>4</td>
      <td>[61.0 &lt; x]</td>
    </tr>
  </tbody>
</table>
</div>



## Saving for later uses


```python
import json

# storing as json file
with open('my_carver.json', 'w') as my_carver_json:
    json.dump(auto_carver.to_json(), my_carver_json)
```

# Feature Selection

### Setting up measures and filters


```python
from AutoCarver.feature_selection import FeatureSelector

n_best = 10  # number of features to select

feature_selector = FeatureSelector(
    quantitative_features=quantitative_features, 
    n_best=n_best,
    pretty_print=True,
)
best_features = feature_selector.select(x_discretized, x_discretized[target])
```

    ------
    [FeatureSelector] Selecting from Features: ['age', 'NumberOfDependents', 'DebtRatio', 'RevolvingUtilizationOfUnsecuredLines', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfOpenCreditLinesAndLoans', 'NumberRealEstateLoansOrLines', 'NumberOfTime30-59DaysPastDueNotWorse', 'MonthlyIncome']
    ---
    
     - Association between X and y
    


<style type="text/css">
#T_524fe_row0_col1, #T_524fe_row1_col1, #T_524fe_row2_col1, #T_524fe_row3_col1, #T_524fe_row4_col1, #T_524fe_row5_col2, #T_524fe_row7_col1, #T_524fe_row8_col1, #T_524fe_row9_col1, #T_524fe_row9_col4 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_524fe_row0_col2 {
  background-color: #b50927;
  color: #f1f1f1;
}
#T_524fe_row0_col4, #T_524fe_row1_col2, #T_524fe_row5_col1 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_524fe_row1_col4 {
  background-color: #f6bfa6;
  color: #000000;
}
#T_524fe_row2_col2 {
  background-color: #c0d4f5;
  color: #000000;
}
#T_524fe_row2_col4 {
  background-color: #e1dad6;
  color: #000000;
}
#T_524fe_row3_col2 {
  background-color: #e36b54;
  color: #f1f1f1;
}
#T_524fe_row3_col4 {
  background-color: #dfdbd9;
  color: #000000;
}
#T_524fe_row4_col2 {
  background-color: #6687ed;
  color: #f1f1f1;
}
#T_524fe_row4_col4 {
  background-color: #5d7ce6;
  color: #f1f1f1;
}
#T_524fe_row5_col4 {
  background-color: #4257c9;
  color: #f1f1f1;
}
#T_524fe_row6_col1 {
  background-color: #6384eb;
  color: #f1f1f1;
}
#T_524fe_row6_col2 {
  background-color: #dedcdb;
  color: #000000;
}
#T_524fe_row6_col4 {
  background-color: #3d50c3;
  color: #f1f1f1;
}
#T_524fe_row7_col2 {
  background-color: #536edd;
  color: #f1f1f1;
}
#T_524fe_row7_col4, #T_524fe_row8_col4 {
  background-color: #3c4ec2;
  color: #f1f1f1;
}
#T_524fe_row8_col2 {
  background-color: #86a9fc;
  color: #f1f1f1;
}
#T_524fe_row9_col2 {
  background-color: #93b5fe;
  color: #000000;
}
</style>
<table id="T_524fe" style='display:inline'>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_524fe_level0_col0" class="col_heading level0 col0" >dtype</th>
      <th id="T_524fe_level0_col1" class="col_heading level0 col1" >pct_nan</th>
      <th id="T_524fe_level0_col2" class="col_heading level0 col2" >pct_mode</th>
      <th id="T_524fe_level0_col3" class="col_heading level0 col3" >mode</th>
      <th id="T_524fe_level0_col4" class="col_heading level0 col4" >kruskal_measure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_524fe_level0_row0" class="row_heading level0 row0" >NumberOfTimes90DaysLate</th>
      <td id="T_524fe_row0_col0" class="data row0 col0" >int64</td>
      <td id="T_524fe_row0_col1" class="data row0 col1" >0.000000</td>
      <td id="T_524fe_row0_col2" class="data row0 col2" >0.943960</td>
      <td id="T_524fe_row0_col3" class="data row0 col3" >0</td>
      <td id="T_524fe_row0_col4" class="data row0 col4" >11854.250713</td>
    </tr>
    <tr>
      <th id="T_524fe_level0_row1" class="row_heading level0 row1" >NumberOfTime60-89DaysPastDueNotWorse</th>
      <td id="T_524fe_row1_col0" class="data row1 col0" >int64</td>
      <td id="T_524fe_row1_col1" class="data row1 col1" >0.000000</td>
      <td id="T_524fe_row1_col2" class="data row1 col2" >0.949254</td>
      <td id="T_524fe_row1_col3" class="data row1 col3" >0</td>
      <td id="T_524fe_row1_col4" class="data row1 col4" >7636.377475</td>
    </tr>
    <tr>
      <th id="T_524fe_level0_row2" class="row_heading level0 row2" >RevolvingUtilizationOfUnsecuredLines</th>
      <td id="T_524fe_row2_col0" class="data row2 col0" >float64</td>
      <td id="T_524fe_row2_col1" class="data row2 col1" >0.000000</td>
      <td id="T_524fe_row2_col2" class="data row2 col2" >0.500000</td>
      <td id="T_524fe_row2_col3" class="data row2 col3" >0.000000</td>
      <td id="T_524fe_row2_col4" class="data row2 col4" >6158.299923</td>
    </tr>
    <tr>
      <th id="T_524fe_level0_row3" class="row_heading level0 row3" >NumberOfTime30-59DaysPastDueNotWorse</th>
      <td id="T_524fe_row3_col0" class="data row3 col0" >int64</td>
      <td id="T_524fe_row3_col1" class="data row3 col1" >0.000000</td>
      <td id="T_524fe_row3_col2" class="data row3 col2" >0.839035</td>
      <td id="T_524fe_row3_col3" class="data row3 col3" >0</td>
      <td id="T_524fe_row3_col4" class="data row3 col4" >6076.412797</td>
    </tr>
    <tr>
      <th id="T_524fe_level0_row4" class="row_heading level0 row4" >age</th>
      <td id="T_524fe_row4_col0" class="data row4 col0" >int64</td>
      <td id="T_524fe_row4_col1" class="data row4 col1" >0.000000</td>
      <td id="T_524fe_row4_col2" class="data row4 col2" >0.303592</td>
      <td id="T_524fe_row4_col3" class="data row4 col3" >1</td>
      <td id="T_524fe_row4_col4" class="data row4 col4" >1370.032527</td>
    </tr>
    <tr>
      <th id="T_524fe_level0_row5" class="row_heading level0 row5" >MonthlyIncome</th>
      <td id="T_524fe_row5_col0" class="data row5 col0" >float64</td>
      <td id="T_524fe_row5_col1" class="data row5 col1" >0.197383</td>
      <td id="T_524fe_row5_col2" class="data row5 col2" >0.200945</td>
      <td id="T_524fe_row5_col3" class="data row5 col3" >2.000000</td>
      <td id="T_524fe_row5_col4" class="data row5 col4" >337.382149</td>
    </tr>
    <tr>
      <th id="T_524fe_level0_row6" class="row_heading level0 row6" >NumberOfDependents</th>
      <td id="T_524fe_row6_col0" class="data row6 col0" >float64</td>
      <td id="T_524fe_row6_col1" class="data row6 col1" >0.026129</td>
      <td id="T_524fe_row6_col2" class="data row6 col2" >0.579741</td>
      <td id="T_524fe_row6_col3" class="data row6 col3" >0.000000</td>
      <td id="T_524fe_row6_col4" class="data row6 col4" >175.384642</td>
    </tr>
    <tr>
      <th id="T_524fe_level0_row7" class="row_heading level0 row7" >NumberOfOpenCreditLinesAndLoans</th>
      <td id="T_524fe_row7_col0" class="data row7 col0" >int64</td>
      <td id="T_524fe_row7_col1" class="data row7 col1" >0.000000</td>
      <td id="T_524fe_row7_col2" class="data row7 col2" >0.261602</td>
      <td id="T_524fe_row7_col3" class="data row7 col3" >2</td>
      <td id="T_524fe_row7_col4" class="data row7 col4" >121.575254</td>
    </tr>
    <tr>
      <th id="T_524fe_level0_row8" class="row_heading level0 row8" >NumberRealEstateLoansOrLines</th>
      <td id="T_524fe_row8_col0" class="data row8 col0" >int64</td>
      <td id="T_524fe_row8_col1" class="data row8 col1" >0.000000</td>
      <td id="T_524fe_row8_col2" class="data row8 col2" >0.375512</td>
      <td id="T_524fe_row8_col3" class="data row8 col3" >0</td>
      <td id="T_524fe_row8_col4" class="data row8 col4" >120.839911</td>
    </tr>
    <tr>
      <th id="T_524fe_level0_row9" class="row_heading level0 row9" >DebtRatio</th>
      <td id="T_524fe_row9_col0" class="data row9 col0" >float64</td>
      <td id="T_524fe_row9_col1" class="data row9 col1" >0.000000</td>
      <td id="T_524fe_row9_col2" class="data row9 col2" >0.400000</td>
      <td id="T_524fe_row9_col3" class="data row9 col3" >0.000000</td>
      <td id="T_524fe_row9_col4" class="data row9 col4" >47.123007</td>
    </tr>
  </tbody>
</table>



    
     - Association between X and y, filtered for inter-feature assocation
    


<style type="text/css">
#T_bfa45_row0_col1, #T_bfa45_row1_col1, #T_bfa45_row2_col1, #T_bfa45_row3_col1, #T_bfa45_row4_col1, #T_bfa45_row5_col2, #T_bfa45_row7_col1, #T_bfa45_row8_col1, #T_bfa45_row9_col1, #T_bfa45_row9_col4 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_bfa45_row0_col2 {
  background-color: #b50927;
  color: #f1f1f1;
}
#T_bfa45_row0_col4, #T_bfa45_row1_col2, #T_bfa45_row5_col1 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_bfa45_row1_col4 {
  background-color: #f6bfa6;
  color: #000000;
}
#T_bfa45_row2_col2 {
  background-color: #c0d4f5;
  color: #000000;
}
#T_bfa45_row2_col4 {
  background-color: #e1dad6;
  color: #000000;
}
#T_bfa45_row3_col2 {
  background-color: #e36b54;
  color: #f1f1f1;
}
#T_bfa45_row3_col4 {
  background-color: #dfdbd9;
  color: #000000;
}
#T_bfa45_row4_col2 {
  background-color: #6687ed;
  color: #f1f1f1;
}
#T_bfa45_row4_col4 {
  background-color: #5d7ce6;
  color: #f1f1f1;
}
#T_bfa45_row5_col4 {
  background-color: #4257c9;
  color: #f1f1f1;
}
#T_bfa45_row6_col1 {
  background-color: #6384eb;
  color: #f1f1f1;
}
#T_bfa45_row6_col2 {
  background-color: #dedcdb;
  color: #000000;
}
#T_bfa45_row6_col4 {
  background-color: #3d50c3;
  color: #f1f1f1;
}
#T_bfa45_row7_col2 {
  background-color: #536edd;
  color: #f1f1f1;
}
#T_bfa45_row7_col4, #T_bfa45_row8_col4 {
  background-color: #3c4ec2;
  color: #f1f1f1;
}
#T_bfa45_row8_col2 {
  background-color: #86a9fc;
  color: #f1f1f1;
}
#T_bfa45_row9_col2 {
  background-color: #93b5fe;
  color: #000000;
}
</style>
<table id="T_bfa45" style='display:inline'>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_bfa45_level0_col0" class="col_heading level0 col0" >dtype</th>
      <th id="T_bfa45_level0_col1" class="col_heading level0 col1" >pct_nan</th>
      <th id="T_bfa45_level0_col2" class="col_heading level0 col2" >pct_mode</th>
      <th id="T_bfa45_level0_col3" class="col_heading level0 col3" >mode</th>
      <th id="T_bfa45_level0_col4" class="col_heading level0 col4" >kruskal_measure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_bfa45_level0_row0" class="row_heading level0 row0" >NumberOfTimes90DaysLate</th>
      <td id="T_bfa45_row0_col0" class="data row0 col0" >int64</td>
      <td id="T_bfa45_row0_col1" class="data row0 col1" >0.000000</td>
      <td id="T_bfa45_row0_col2" class="data row0 col2" >0.943960</td>
      <td id="T_bfa45_row0_col3" class="data row0 col3" >0</td>
      <td id="T_bfa45_row0_col4" class="data row0 col4" >11854.250713</td>
    </tr>
    <tr>
      <th id="T_bfa45_level0_row1" class="row_heading level0 row1" >NumberOfTime60-89DaysPastDueNotWorse</th>
      <td id="T_bfa45_row1_col0" class="data row1 col0" >int64</td>
      <td id="T_bfa45_row1_col1" class="data row1 col1" >0.000000</td>
      <td id="T_bfa45_row1_col2" class="data row1 col2" >0.949254</td>
      <td id="T_bfa45_row1_col3" class="data row1 col3" >0</td>
      <td id="T_bfa45_row1_col4" class="data row1 col4" >7636.377475</td>
    </tr>
    <tr>
      <th id="T_bfa45_level0_row2" class="row_heading level0 row2" >RevolvingUtilizationOfUnsecuredLines</th>
      <td id="T_bfa45_row2_col0" class="data row2 col0" >float64</td>
      <td id="T_bfa45_row2_col1" class="data row2 col1" >0.000000</td>
      <td id="T_bfa45_row2_col2" class="data row2 col2" >0.500000</td>
      <td id="T_bfa45_row2_col3" class="data row2 col3" >0.000000</td>
      <td id="T_bfa45_row2_col4" class="data row2 col4" >6158.299923</td>
    </tr>
    <tr>
      <th id="T_bfa45_level0_row3" class="row_heading level0 row3" >NumberOfTime30-59DaysPastDueNotWorse</th>
      <td id="T_bfa45_row3_col0" class="data row3 col0" >int64</td>
      <td id="T_bfa45_row3_col1" class="data row3 col1" >0.000000</td>
      <td id="T_bfa45_row3_col2" class="data row3 col2" >0.839035</td>
      <td id="T_bfa45_row3_col3" class="data row3 col3" >0</td>
      <td id="T_bfa45_row3_col4" class="data row3 col4" >6076.412797</td>
    </tr>
    <tr>
      <th id="T_bfa45_level0_row4" class="row_heading level0 row4" >age</th>
      <td id="T_bfa45_row4_col0" class="data row4 col0" >int64</td>
      <td id="T_bfa45_row4_col1" class="data row4 col1" >0.000000</td>
      <td id="T_bfa45_row4_col2" class="data row4 col2" >0.303592</td>
      <td id="T_bfa45_row4_col3" class="data row4 col3" >1</td>
      <td id="T_bfa45_row4_col4" class="data row4 col4" >1370.032527</td>
    </tr>
    <tr>
      <th id="T_bfa45_level0_row5" class="row_heading level0 row5" >MonthlyIncome</th>
      <td id="T_bfa45_row5_col0" class="data row5 col0" >float64</td>
      <td id="T_bfa45_row5_col1" class="data row5 col1" >0.197383</td>
      <td id="T_bfa45_row5_col2" class="data row5 col2" >0.200945</td>
      <td id="T_bfa45_row5_col3" class="data row5 col3" >2.000000</td>
      <td id="T_bfa45_row5_col4" class="data row5 col4" >337.382149</td>
    </tr>
    <tr>
      <th id="T_bfa45_level0_row6" class="row_heading level0 row6" >NumberOfDependents</th>
      <td id="T_bfa45_row6_col0" class="data row6 col0" >float64</td>
      <td id="T_bfa45_row6_col1" class="data row6 col1" >0.026129</td>
      <td id="T_bfa45_row6_col2" class="data row6 col2" >0.579741</td>
      <td id="T_bfa45_row6_col3" class="data row6 col3" >0.000000</td>
      <td id="T_bfa45_row6_col4" class="data row6 col4" >175.384642</td>
    </tr>
    <tr>
      <th id="T_bfa45_level0_row7" class="row_heading level0 row7" >NumberOfOpenCreditLinesAndLoans</th>
      <td id="T_bfa45_row7_col0" class="data row7 col0" >int64</td>
      <td id="T_bfa45_row7_col1" class="data row7 col1" >0.000000</td>
      <td id="T_bfa45_row7_col2" class="data row7 col2" >0.261602</td>
      <td id="T_bfa45_row7_col3" class="data row7 col3" >2</td>
      <td id="T_bfa45_row7_col4" class="data row7 col4" >121.575254</td>
    </tr>
    <tr>
      <th id="T_bfa45_level0_row8" class="row_heading level0 row8" >NumberRealEstateLoansOrLines</th>
      <td id="T_bfa45_row8_col0" class="data row8 col0" >int64</td>
      <td id="T_bfa45_row8_col1" class="data row8 col1" >0.000000</td>
      <td id="T_bfa45_row8_col2" class="data row8 col2" >0.375512</td>
      <td id="T_bfa45_row8_col3" class="data row8 col3" >0</td>
      <td id="T_bfa45_row8_col4" class="data row8 col4" >120.839911</td>
    </tr>
    <tr>
      <th id="T_bfa45_level0_row9" class="row_heading level0 row9" >DebtRatio</th>
      <td id="T_bfa45_row9_col0" class="data row9 col0" >float64</td>
      <td id="T_bfa45_row9_col1" class="data row9 col1" >0.000000</td>
      <td id="T_bfa45_row9_col2" class="data row9 col2" >0.400000</td>
      <td id="T_bfa45_row9_col3" class="data row9 col3" >0.000000</td>
      <td id="T_bfa45_row9_col4" class="data row9 col4" >47.123007</td>
    </tr>
  </tbody>
</table>



    ------
    
    




    ['NumberOfTimes90DaysLate',
     'NumberOfTime60-89DaysPastDueNotWorse',
     'RevolvingUtilizationOfUnsecuredLines',
     'NumberOfTime30-59DaysPastDueNotWorse',
     'age',
     'MonthlyIncome',
     'NumberOfDependents',
     'NumberOfOpenCreditLinesAndLoans',
     'NumberRealEstateLoansOrLines',
     'DebtRatio']



### Enjoy modeling!
