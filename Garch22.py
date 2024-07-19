from random import gauss
import matplotlib.pyplot as plt
import numpy as np
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# using the arch library for convenience; the parameters are known to us:
# a_t = \epsilon_t \sqrt{\omega + \alpha_1 a^2_{t-1} + \alpha_2 a^2_{t-2} + \beta_1 \sigma^2_{t-1} + \beta_2 \sigma^2_{t-2}}

# sample dataset creation
n = 1000
omega = 0.5

alpha_1 = 0.15
alpha_2 = 0.25

beta_1 = 0.3
beta_2 = 0.4

test_size = int(n*0.1)

# normal distribution for the time-series, and initial values for the volatility
series = [gauss(0,1), gauss(0,1)]
vols = [1, 1]

for _ in range(n):
    new_vol = np.sqrt(omega + alpha_1*series[-1]**2 + alpha_2*series[-2]**2 + beta_1*vols[-1]**2 + beta_2*vols[-2]**2)
    new_val = gauss(0,1) * new_vol
    
    vols.append(new_vol)
    series.append(new_val)

# series and volatility plots
plt.figure(figsize=(10,4))
plt.plot(series)
plot_pacf(np.array(series)**2)
plt.title('Simulated GARCH(2,2) Data', fontsize=20)
#plt.show()
plt.savefig('./series_Garch_22.png')

plt.figure(figsize=(10,4))
plt.plot(vols)
plt.title('Data Volatility', fontsize=20)
plt.savefig('./volatility_Garch_22.png')

plt.figure(figsize=(10,4))
plt.plot(series)
plt.plot(vols, color='red')
plt.title('Data and Volatility', fontsize=20)
plt.savefig('./seriesnvol_Garch_22.png')

print(len(series), len(vols))


# Fitting and Prediction of the model 

train, test = series[:-test_size], series[-test_size:]
model = arch_model(train, p=2, q=2)

model_fit = model.fit()
model_fit.summary()

predictions = model_fit.forecast(horizon=test_size)

plt.figure(figsize=(9,4.5))
true, = plt.plot(vols[-test_size:])
preds, = plt.plot(np.sqrt(predictions.variance.values[-1, :]))
plt.title('Volatility Prediction', fontsize=20)
plt.legend(['True Volatility', 'Predicted Volatility'], fontsize=16)
plt.savefig('Volatility_prediction_Garch22.png')

predictions_long_term = model_fit.forecast(horizon=1000)
plt.figure(figsize=(9,4.5))
true, = plt.plot(vols[-test_size:])
preds, = plt.plot(np.sqrt(predictions_long_term.variance.values[-1, :]))
plt.title('Long Term Volatility Prediction', fontsize=20)
plt.legend(['True Volatility', 'Predicted Volatility'], fontsize=16)
plt.savefig('Truevspredicted_Volatility_Garch22.png')

# WE use rolling predictions with a small rollover to get a much better fit
rolling_predictions = []
for i in range(test_size):
    train = series[:-(test_size-i)]
    model = arch_model(train, p=2, q=2)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))

plt.figure(figsize=(10,4))
true, = plt.plot(vols[-test_size:])
preds, = plt.plot(rolling_predictions)
plt.title('Volatility Prediction - Rolling Forecast', fontsize=20)
plt.legend(['True Volatility', 'Predicted Volatility'], fontsize=16)
plt.show()
plt.savefig('Volatility_prediction_rollingforecast_Garch22.png')



