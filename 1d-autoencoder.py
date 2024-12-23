
#%%
# ###### Source: https://www.kaggle.com/code/cnumber/neurips-ariel-data-challenge-2024-final-submission

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import joblib

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import itertools

from scipy.optimize import minimize
from scipy import optimize

from astropy.stats import sigma_clip

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

wd = f"{}/datasets/arial-data-challenge-2024"
dataset = 'train'
adc_info = pd.read_csv(f'{wd}/{dataset}_adc_info.csv',index_col='planet_id')
# # axis_info = pd.read_parquet(f'{wd}/axis_info.parquet')

# if dataset == "train":
#     adc_info = adc_info[:10]

train_targets = np.loadtxt(f'{wd}/train_labels.csv', delimiter=',', skiprows=1)
targets = train_targets[:,1:-1]

#%%
def apply_linear_corr(linear_corr,clean_signal):
    linear_corr = np.flip(linear_corr, axis=0)
    for x, y in itertools.product(
                range(clean_signal.shape[1]), range(clean_signal.shape[2])
            ):
        poli = np.poly1d(linear_corr[:, x, y])
        clean_signal[:, x, y] = poli(clean_signal[:, x, y])
    return clean_signal

def clean_dark(signal, dark, dt):
    dark = np.tile(dark, (signal.shape[0], 1, 1))
    signal -= dark* dt[:, np.newaxis, np.newaxis]
    return signal

def preproc(dataset, adc_info, sensor, binning = 15, num_files=10, target=None):
    cut_inf, cut_sup = 39, 321
    sensor_sizes_dict = {"AIRS-CH0":[[11250, 32, 356], [1, 32, cut_sup-cut_inf]], "FGS1":[[135000, 32, 32], [1, 32, 32]]}
    binned_dict = {"AIRS-CH0":[11250 // binning // 2, 282], "FGS1":[135000 // binning // 2]}
    linear_corr_dict = {"AIRS-CH0":(6, 32, 356), "FGS1":(6, 32, 32)}
    planet_ids = adc_info.index
    
    targets = []
    feats = []
    for i, planet_id in tqdm(list(enumerate(planet_ids))):
        try: 
            signal = pd.read_parquet(f'{wd}/{dataset}/{planet_id}/{sensor}_signal.parquet', engine='pyarrow').to_numpy()
            dark_frame = pd.read_parquet(f'{wd}/{dataset}/' + str(planet_id) + '/' + sensor + '_calibration/dark.parquet', engine='pyarrow').to_numpy()
            dead_frame = pd.read_parquet(f'{wd}/{dataset}/' + str(planet_id) + '/' + sensor + '_calibration/dead.parquet', engine='pyarrow').to_numpy()
            flat_frame = pd.read_parquet(f'{wd}/{dataset}/' + str(planet_id) + '/' + sensor + '_calibration/flat.parquet', engine='pyarrow').to_numpy()
            linear_corr = pd.read_parquet(f'{wd}/{dataset}/' + str(planet_id) + '/' + sensor + '_calibration/linear_corr.parquet').values.astype(np.float64).reshape(linear_corr_dict[sensor])

            signal = signal.reshape(sensor_sizes_dict[sensor][0]) 
            gain = adc_info[f'{sensor}_adc_gain'].values[i]
            offset = adc_info[f'{sensor}_adc_offset'].values[i]
            signal = signal / gain + offset
            
            hot = sigma_clip(
                dark_frame, sigma=5, maxiters=5
            ).mask
            
            if sensor != "FGS1":
                signal = signal[:, :, cut_inf:cut_sup] 
                dt = np.ones(len(signal))*0.1 
                dt[1::2] += 4.5 #@bilzard idea
                linear_corr = linear_corr[:, :, cut_inf:cut_sup]
                dark_frame = dark_frame[:, cut_inf:cut_sup]
                dead_frame = dead_frame[:, cut_inf:cut_sup]
                flat_frame = flat_frame[:, cut_inf:cut_sup]
                hot = hot[:, cut_inf:cut_sup]
            else:
                dt = np.ones(len(signal))*0.1
                dt[1::2] += 0.1
                
            # signal = signal.clip(0) #@graySnow idea
            linear_corr_signal = apply_linear_corr(linear_corr, signal)
            signal = clean_dark(linear_corr_signal, dark_frame, dt)
            
            flat = flat_frame.reshape(sensor_sizes_dict[sensor][1])
            flat[dead_frame.reshape(sensor_sizes_dict[sensor][1])] = np.nan
            #flat[hot.reshape(sensor_sizes_dict[sensor][1])] = np.nan
            signal = signal / flat
                
            if sensor == "FGS1":
                signal = signal[:,10:22,10:22] # **** updates ****
                signal = signal.reshape(sensor_sizes_dict[sensor][0][0],144) # # **** updates ****

            if sensor != "FGS1":
                # backgrounds are [0:8] and [24:32]
                signal_bg = np.nanmean(
                    np.concatenate([signal[:, 0:8, :], signal[:, 24:32, :]], axis=1), axis=1
                )
                signal_bg[np.isnan(signal_bg)] = 0
                signal = signal[:, 8:24, :]  # **** updates ****

            mean_signal = np.nanmean(signal, axis=1)
            cds_signal = mean_signal[1::2] - mean_signal[0::2]
            cds_signal_bg = signal_bg[1::2] - signal_bg[0::2]

            cds_signal_bg = np.nanmean(cds_signal_bg, axis=0, keepdims=True)

            cds_signal -= cds_signal_bg

            binned = np.zeros((binned_dict[sensor]))
            for j in range(cds_signal.shape[0] // binning):
                binned[j] = cds_signal[j * binning : j * binning + binning].mean(axis=0)

            if sensor == "FGS1":
                binned = binned.reshape((binned.shape[0], 1))

            feats.append(binned)
            targets.append(target[i])
            if len(feats) == num_files:
                break

        except Exception as e:
            print(f"Unexpected error: {e}")
    
    return np.stack(feats), np.stack(targets)

inputs_np, targets_np  = preproc(f'{dataset}', adc_info, "AIRS-CH0", 1, num_files=len(adc_info), target=targets)
np.savez('preprocessed_data.npz', input=inputs_np, target=targets_np)
# inputs_np, targets_np = preproc(f'{dataset}', adc_info, "AIRS-CH0", 1, num_files=5,  target=targets)

#%%
class Autoencoder(nn.Module):#best is 2 layers each with 128 and half
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim//2),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim//2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, input_dim),
            nn.Tanh() #best soln
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

batch_size = 128
input_shape = (5625, 282)
model = Autoencoder(input_shape[1], latent_dim=128)

#%%
loaded =np.load("preprocessed_data.npz")
targets_np, inputs_np = loaded['target'], loaded['input']

flat_input = inputs_np.mean(axis=1)
# norm = (flat_input - flat_input.mean(axis=0)) / flat_input.std(axis=0)
norm = (flat_input - flat_input.mean()) / flat_input.std() #best
norm = torch.tensor(norm).unsqueeze(1).float()

targets_tensor = torch.tensor(targets_np).unsqueeze(1).float() 
targets_tensor = targets_tensor/ targets_tensor.max(dim=0).values #best
# targets_tensor = (targets_tensor-targets_tensor.mean(axis=0))/targets_tensor.std(axis=0)

dataset = TensorDataset(norm, targets_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

criterion = nn.MSELoss()
L1_loss = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

model = model.to(device)


#%%
num_epochs = 500
train_losses, val_losses = [], []
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for step, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets) #+ L1_loss(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    epoch_loss= total_loss / (step+1)
    if epoch % 10 == 0:
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        model.eval()
        total_loss = 0
        sample_gt, sample_pred = [], []
        with torch.no_grad():
            for step, (inputs, targets) in enumerate(tqdm(val_loader)):
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets) 

                total_loss += loss.item()

                sample_gt.append(targets)
                sample_pred.append(outputs)
            
            epoch_loss= total_loss / (step+1)
            val_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {epoch_loss:.4f}")


#%%
print("pause")

rows, cols = 4, 4
fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
axes = axes.flatten()
delta = 0
for i in range(rows * cols):
    delta = sample_pred[i][0, 0, :].detach().cpu().numpy()-sample_gt[i][0, 0, :].detach().cpu().numpy()
    delta = delta.mean()
    delta=0
    axes[i].plot(sample_gt[i][0, 0, :].detach().cpu().numpy()+delta, label='Target', color='blue')
    axes[i].plot(sample_pred[i][0, 0, :].detach().cpu().numpy(), label='Prediction', color='red')
    axes[i].legend()
    axes[i].set_title(f"Sample {i+1}, {sample_gt[i][0, 0, :].mean():.4f}, {sample_pred[i][0, 0, :].mean():.4f}")
plt.tight_layout()
plt.savefig('test_grid.png')
# %%

rows, cols = 4, 4
fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
axes = axes.flatten()
for i in range(rows * cols):
    axes[i].plot(sample_gt[i][0, 0, :].detach().cpu().numpy()-sample_pred[i][0, 0, :].detach().cpu().numpy(), label='Target', color='blue')
    axes[i].set_title(f"Sample {i+1}, {sample_gt[i][0, 0, :].mean():.4f}, {sample_pred[i][0, 0, :].mean():.4f}")
plt.tight_layout()
plt.savefig('test_grid_difference.png')