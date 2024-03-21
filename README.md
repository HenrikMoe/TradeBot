# TradeBot - Numerical Regressive Model Set + Implementation 

## Set Env & Run Models

Boilerplate Numerical/Regressive Model: Chat3deriv.py

startup:
Navigate to the directory where you want to create the virtual environment
```linux
# cd (project root) 
```
Create a virtual environment (you can choose any name, here we use "venv")
```linux
python3 -m venv venv
```

Activate the virtual environment
```linux
source venv/bin/activate
```

Install required packages
```linux
pip install numpy TFANN matplotlib scikit-learn tensorflow
```

Make sure you are in the directory containing your script
```linux
cd (rn the csv and the py file are in Machine)
```
Run the script
```linux
python chat3deriv.py
```

manually add the csv file to the venv if needed

+ Add sourced method training configs

+ Add JAX training config and methods.

# Unmethodized Configs

## Config 1 

Network Type:
Characterstic N: 
(continue to classify)
n 32 by 32
.001 learn rate

Test Results:

got 69,440.44 versus 70,083
from range 66230 bottom. -------- mid

then 70,370445 versus 68673.05469 ----- bad 
from 68053.125

then 69,96009 versus 70005.20313 ---- good
from 68239.97656

then 69,91726 versus 72850.71094 --- bad 
frmom 67194.88281

then 69,47696 versus 72487.10156 ---- bad 
from 69210.07813

then 69.8324 versus x

## Config 2 

Network Type:
Characterstic N: 
(continue to classify)
n 32 by 32
trial @ 0.005 learn rate: 

got 69,0288 versus 70,083
from range 66230 bottom. -------- mid

then 70,370445 versus 68673.05469 ----- bad 
from 68053.125

then 69,96009 versus 70005.20313 ---- good
from 68239.97656

then 69,91726 versus 72850.71094 --- bad 
frmom 67194.88281

then 69,47696 versus 72487.10156 ---- bad 
from 69210.07813


## Tuning Overview Notes: 

Scaling/Normalization:
Ensure that the scaling factors used for training are applied consistently during prediction. In a real-world scenario, you would save the scaling factors during training and use them for prediction. Here, I assumed that the range of scaling factors is similar for both training and prediction. It's essential to apply the same scaling to input data during both phases.
python
Copy code
### Use the same scaling factors for both training and prediction
### Note: In a real-world scenario, you would save the scaling factors during training and use them for prediction.
### For simplicity, I'm assuming here that the range of the scaling factors is similar in both cases.
A_scaled = scale(A)
Imputation for NaN values:
You've used np.nan_to_num for handling NaN values, replacing them with the mean. Depending on your data, you might want to consider more sophisticated imputation techniques, such as using the mean or median of the column. However, this depends on the nature of your data and the impact of imputation on the model.
python
Copy code
### Handle NaN values with more sophisticated imputation
low_prices = np.nan_to_num(low_prices, nan=np.nanmean(low_prices))
high_prices = np.nan_to_num(high_prices, nan=np.nanmean(high_prices))
y = np.nan_to_num(y, nan=np.nanmean(y))
Adjust Learning Rate:
Experiment with different learning rates to find an optimal value. This can significantly impact the training process. You've set the learning rate to 0.001, and you can adjust it based on trial and error.
python
Copy code
### Recreate optimizer instance with a constant learning rate
learning_rate = 0.001  # You can adjust this value if needed
self.optmzr = tf.optimizers.Adam(learning_rate=learning_rate)
Additional Hyperparameter Tuning:

Consider experimenting with other hyperparameters like the number of hidden layers, the number of neurons in each layer, and the regularization strength to find a combination that works well for your specific dataset.
Verbose Output:

You have a verbose option in your model. If you want to see training progress details, set verbose=True when creating the SeqMLP instance.
python
Copy code
### Create the custom SeqMLP instance with verbose output
seq_mlp = SeqMLP(input_size=2, hidden_size=[32, 32], output_size=1, actvFn='tanh', learnRate=0.001, maxItr=2000, tol=1e-2, verbose=True, reg=0.001)
Cross-Validation:
Consider using cross-validation to get a more robust evaluation of your model's performance across different hold-out periods.
These suggestions aim to help you fine-tune your model and improve its generalization across various scenarios.


## System Development Path

Trading API - stock exchange clients

+ Alpaca JAVA - https://github.com/Petersoj/alpaca-java

++ Alpaca Infrastructure 

- get account credentials 

Stripe for deposits and withdrawls ? - direct to alpaca?

Roth IRA infrasturcutre?

AI Predicition Algo:

+ https://github.com/yazanobeidi/flow

-- rsi and macd indicators: https://github.com/yazanobeidi/flow/blob/master/python/indicators.py

- run simulation

+ https://github.com/crypto-code/Stock-Market-AI-GUI

+ https://github.com/PyPatel/Machine-Learning-and-AI-in-Trading/tree/master

- master TF file - https://github.com/PyPatel/Machine-Learning-and-AI-in-Trading/blob/master/Trading_MLP_TF.py


Connect Alpaca to the prediction read in for all stock prices per 12h for forever for an asset, verfiy csv format with the TF model.

Algo returns prediction.

We code some logic to buy and sell the stock given our protfolio value - some optimzition function that we can hard code. - or already included





