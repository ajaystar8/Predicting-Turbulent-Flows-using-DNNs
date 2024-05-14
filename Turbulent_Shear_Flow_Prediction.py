from keras.layers import Dense, LSTM
from keras.models import Sequential, load_model

from config import *
from data_generator import *
from visualize import *

u2x = np.zeros((len(x), len(y), len(z)))
u1x = np.zeros((len(x), len(y), len(z)))
u3y = np.zeros((len(x), len(y), len(z)))
u3z = np.zeros((len(x), len(y), len(z)))
u4z = np.zeros((len(x), len(y), len(z)))
u5z = np.zeros((len(x), len(y), len(z)))
u6x = np.zeros((len(x), len(y), len(z)))
u6z = np.zeros((len(x), len(y), len(z)))
u7x = np.zeros((len(x), len(y), len(z)))
u7z = np.zeros((len(x), len(y), len(z)))
u8x = np.zeros((len(x), len(y), len(z)))
u8y = np.zeros((len(x), len(y), len(z)))
u8z = np.zeros((len(x), len(y), len(z)))
u9x = np.zeros((len(x), len(y), len(z)))

for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            u1x[i, j, k] = (np.sqrt(2) * np.sin(np.pi * y[j] / 2.))
            u2x[i, j, k] = (4 / np.sqrt(3)) * \
                           (np.cos(np.pi * y[j] / 2)) ** 2 * np.cos(g * z[k])
            u3y[i, j, k] = (2. / np.sqrt(4 * g ** 2 + np.pi ** 2)) * 2 * g * \
                           np.cos(np.pi * y[j] / 2) * np.cos(g * z[k])
            u3z[i, j, k] = (2. / np.sqrt(4 * g ** 2 + np.pi ** 2)) * np.pi * \
                           np.sin(np.pi * y[j] / 2) * np.sin(g * z[k])
            u4z[i, j, k] = (4 / np.sqrt(3)) * np.cos(a * x[i]) * \
                           (np.cos(np.pi * y[j] / 2)) ** 2
            u5z[i, j, k] = 2 * np.sin(a * x[i]) * np.sin(np.pi * y[j] / 2)
            u6x[i, j, k] = (4 * np.sqrt(2) / np.sqrt(3 * (a ** 2 + g ** 2))) * (-g) * \
                           np.cos(a * x[i]) * (np.cos(np.pi * y[j] / 2)) ** 2 * np.sin(g * z[k])
            u6z[i, j, k] = (4 * np.sqrt(2) / np.sqrt(3 * (a ** 2 + g ** 2))) * a * \
                           np.sin(a * x[i]) * (np.cos(np.pi * y[j] / 2)) ** 2 * np.cos(g * z[k])
            u7x[i, j, k] = (2 * np.sqrt(2) / np.sqrt(a ** 2 + g ** 2)) * g * \
                           np.sin(a * x[i]) * np.sin(np.pi * y[j] / 2) * np.sin(g * z[k])
            u7z[i, j, k] = (2 * np.sqrt(2) / np.sqrt(a ** 2 + g ** 2)) * a * \
                           np.cos(a * x[i]) * np.sin(np.pi * y[j] / 2) * np.cos(g * z[k])
            u8x[i, j, k] = (2 * np.sqrt(2) / (np.sqrt((a ** 2 + g ** 2) * (4 * a ** 2 + 4 * g ** 2 + np.pi ** 2)))
                            ) * np.pi * a * np.sin(a * x[i]) * np.sin(np.pi * y[j] / 2) * np.sin(g * z[k])
            u8y[i, j, k] = (2 * np.sqrt(2) / (np.sqrt((a ** 2 + g ** 2) * (4 * a ** 2 + 4 * g ** 2 + np.pi ** 2)))) * \
                           2 * (a ** 2 + g ** 2) * np.cos(a * x[i]) * \
                           np.cos(np.pi * y[j] / 2) * np.sin(g * z[k])
            u8z[i, j, k] = (2 * np.sqrt(2) / (np.sqrt((a ** 2 + g ** 2) * (4 * a ** 2 + 4 * g ** 2 + np.pi ** 2)))) * \
                           (-np.pi * g) * np.cos(a * x[i]) * np.sin(np.pi * y[j] / 2) * np.cos(g * z[k])
            u9x[i, j, k] = np.sqrt(2) * np.sin(3 * np.pi * y[j] / 2)

datasets, init_params = generate_datasets(12)

train_init_params = init_params[:10]
test_init_params = init_params[10:]

train_data = datasets[:10]
test_data = datasets[10:]

np.save('train_init_params.npy', train_init_params)
np.save('test_init_params.npy', test_init_params)

np.save('train_data.npy', train_data)
np.save('test_data.npy', test_data)

trained_model = False

if trained_model:
    model = load_model('path/to/trained/model')
else:
    NUM_EPOCHS = 10
    model = Sequential()

    model.add(LSTM(HIDDEN_UNITS,
                   input_shape=(LOOK_BACK, 9),
                   kernel_initializer='glorot_normal',
                   return_sequences=False))

    model.add(Dense(9, activation='tanh'))
    model.compile(optimizer='adam', loss='mean_squared_error')

generate_sequences(test_data, model)

A = np.load('./train_data.npy')

# The number of training samples
nSamples = A.shape[0] * (A.shape[1] - seqLen)

# Initialize the training inputs and outputs using empty arrays
X = np.empty([nSamples, seqLen, A.shape[2]])
Y = np.empty([nSamples, A.shape[2]])

# Fill the input and output arrays with data
k = 0
for i in np.arange(A.shape[0]):
    for j in np.arange(A.shape[1] - seqLen):
        X[k] = A[i, j:j + seqLen]
        Y[k] = A[i, j + seqLen]
        k = k + 1

score = model.fit(X, Y, batch_size=32, epochs=NUM_EPOCHS,
                  verbose=1, validation_split=0.20, shuffle=True)

# Save losses
lossHistory = score.history['loss']
valLossHistory = score.history['val_loss']

# Save the losses
with open('loss_history.txt', 'w') as file:
    for loss, val_loss in zip(lossHistory, valLossHistory):
        file.write(f'{loss}\t{val_loss}\n')

# Save the model
model.save('trained_model' + '.h5')

seq = np.load('./Sequences/series_1.npz')
testSequences = seq['testSeq']
predSequences = seq['predSeq']

atest_ = testSequences
apred_ = predSequences

start_t = 500
end_t = 500


def calculate_components_new(a):
    ux = a[0] * u1x + a[1] * u2x + a[5] * u6x + a[6] * u7x + a[7] * u8x + a[8] * u9x
    uy = a[2] * u3y + a[7] * u8y
    uz = a[2] * u3z + a[3] * u4z + a[4] * u5z + a[5] * u6z + a[6] * u7z + a[7] * u8z
    return ux, uy, uz


for ti in range(start_t, end_t + 1):
    atest = atest_[ti, :]
    apred = apred_[ti, :]

    ux_test, uy_test, uz_test = calculate_components_new(atest)
    ux_pred, uy_pred, uz_pred = calculate_components_new(apred)

    quiver_contour_plots_xavg(ux_test, uy_test, uz_test, ti)
    uxbar_vs_y_plot(ux_pred, ux_test)

magnitude_vs_time_plot(testSequences, predSequences)

atest_ = testSequences

start_t = 500
end_t = start_t

for ti in range(start_t, end_t + 1):
    atest = atest_[ti, :]

    ux_test, uy_test, uz_test = calculate_components_new(atest)

    quiver_contour_plots_xavg(ux_test, uy_test, uz_test, ti)

atest_ = testSequences

start_t = 2000
end_t = start_t

for ti in range(start_t, end_t + 1):
    atest = atest_[ti, :]

    ux_test, uy_test, uz_test = calculate_components_new(atest)

    quiver_contour_plots_xavg(ux_test, uy_test, uz_test, ti)
