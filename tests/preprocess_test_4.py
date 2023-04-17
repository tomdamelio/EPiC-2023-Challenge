#%%
import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import interpolate, signal
from scipy.signal import butter, lfilter, freqz
import neurokit2 as nk
from biosppy.signals import ecg
import biosppy
import numpy as np  
from hrvanalysis import remove_outliers
from hrvanalysis import remove_ectopic_beats
from hrvanalysis import interpolate_nan_values
from matplotlib import pyplot as plt
import scipy
from scipy import interpolate
from biosppy.signals import ecg
import numpy as np
import cvxopt as cv
import cvxopt.solvers
import biosppy
import numpy as np  
from hrvanalysis import remove_outliers
from hrvanalysis import remove_ectopic_beats
from hrvanalysis import interpolate_nan_values
from matplotlib import pyplot as plt
from scipy import interpolate



#%%
#%%
def load_train_data(sub, video, scenario):
    # create file name based on parameters
    file_name = f"sub_{sub}_vid_{video}.csv"

    # load data files
    train_physiology = pd.read_csv(Path(f"../data/raw/scenario_{scenario}/train/physiology", file_name), index_col="time")
    train_annotations = pd.read_csv(Path(f"../data/raw/scenario_{scenario}/train/annotations", file_name), index_col="time")

    return train_physiology, train_annotations

sub = 9
video = 1
scenario = 1

train_physiology, train_annotations = load_train_data(sub, video, scenario)

#%%


def cvxEDA_pyEDA(y, delta, tau0=2., tau1=0.7, delta_knot=10., alpha=8e-4, gamma=1e-2,
           solver=None, options={'reltol':1e-9}):
    """CVXEDA Convex optimization approach to electrodermal activity processing
    This function implements the cvxEDA algorithm described in "cvxEDA: a
    Convex Optimization Approach to Electrodermal Activity Processing"
    (http://dx.doi.org/10.1109/TBME.2015.2474131, also available from the
    authors' homepages).
    Arguments:
       y: observed EDA signal (we recommend normalizing it: y = zscore(y))
       delta: sampling interval (in seconds) of y
       tau0: slow time constant of the Bateman function
       tau1: fast time constant of the Bateman function
       delta_knot: time between knots of the tonic spline function
       alpha: penalization for the sparse SMNA driver
       gamma: penalization for the tonic spline coefficients
       solver: sparse QP solver to be used, see cvxopt.solvers.qp
       options: solver options, see:
                http://cvxopt.org/userguide/coneprog.html#algorithm-parameters
    Returns (see paper for details):
       r: phasic component
       p: sparse SMNA driver of phasic component
       t: tonic component
       l: coefficients of tonic spline
       d: offset and slope of the linear drift term
       e: model residuals
       obj: value of objective function being minimized (eq 15 of paper)
    """
    import numpy as np
    import cvxopt as cv
    import cvxopt.solvers
    
    n = len(y)
    y = cv.matrix(y)

    # bateman ARMA model
    a1 = 1./min(tau1, tau0) # a1 > a0
    a0 = 1./max(tau1, tau0)
    ar = np.array([(a1*delta + 2.) * (a0*delta + 2.), 2.*a1*a0*delta**2 - 8.,
        (a1*delta - 2.) * (a0*delta - 2.)]) / ((a1 - a0) * delta**2)
    ma = np.array([1., 2., 1.])

    # matrices for ARMA model
    i = np.arange(2, n)
    A = cv.spmatrix(np.tile(ar, (n-2,1)), np.c_[i,i,i], np.c_[i,i-1,i-2], (n,n))
    M = cv.spmatrix(np.tile(ma, (n-2,1)), np.c_[i,i,i], np.c_[i,i-1,i-2], (n,n))

    # spline
    delta_knot_s = int(round(delta_knot / delta))
    spl = np.r_[np.arange(1.,delta_knot_s), np.arange(delta_knot_s, 0., -1.)] # order 1
    spl = np.convolve(spl, spl, 'full')
    spl /= max(spl)
    # matrix of spline regressors
    i = np.c_[np.arange(-(len(spl)//2), (len(spl)+1)//2)] + np.r_[np.arange(0, n, delta_knot_s)]
    nB = i.shape[1]
    j = np.tile(np.arange(nB), (len(spl),1))
    p = np.tile(spl, (nB,1)).T
    valid = (i >= 0) & (i < n)
    B = cv.spmatrix(p[valid], i[valid], j[valid])

    # trend
    C = cv.matrix(np.c_[np.ones(n), np.arange(1., n+1.)/n])
    nC = C.size[1]

    # Solve the problem:
    # .5*(M*q + B*l + C*d - y)^2 + alpha*sum(A,1)*p + .5*gamma*l'*l
    # s.t. A*q >= 0

    old_options = cv.solvers.options.copy()
    cv.solvers.options.clear()
    cv.solvers.options.update(options)
    if solver == 'conelp':
        # Use conelp
        z = lambda m,n: cv.spmatrix([],[],[],(m,n))
        G = cv.sparse([[-A,z(2,n),M,z(nB+2,n)],[z(n+2,nC),C,z(nB+2,nC)],
                    [z(n,1),-1,1,z(n+nB+2,1)],[z(2*n+2,1),-1,1,z(nB,1)],
                    [z(n+2,nB),B,z(2,nB),cv.spmatrix(1.0, range(nB), range(nB))]])
        h = cv.matrix([z(n,1),.5,.5,y,.5,.5,z(nB,1)])
        c = cv.matrix([(cv.matrix(alpha, (1,n)) * A).T,z(nC,1),1,gamma,z(nB,1)])
        res = cv.solvers.conelp(c, G, h, dims={'l':n,'q':[n+2,nB+2],'s':[]})
        obj = res['primal objective']
    else:
        # Use qp
        Mt, Ct, Bt = M.T, C.T, B.T
        H = cv.sparse([[Mt*M, Ct*M, Bt*M], [Mt*C, Ct*C, Bt*C], 
                    [Mt*B, Ct*B, Bt*B+gamma*cv.spmatrix(1.0, range(nB), range(nB))]])
        f = cv.matrix([(cv.matrix(alpha, (1,n)) * A).T - Mt*y,  -(Ct*y), -(Bt*y)])
        res = cv.solvers.qp(H, f, cv.spmatrix(-A.V, A.I, A.J, (n,len(f))),
                            cv.matrix(0., (n,1)), solver=solver)
        obj = res['primal objective'] + .5 * (y.T * y)
    cv.solvers.options.clear()
    cv.solvers.options.update(old_options)

    l = res['x'][-nB:]
    d = res['x'][n:n+nC]
    t = B*l + C*d
    q = res['x'][:n]
    p = A * q
    r = M * q
    e = y - r - t

    return (np.array(a).ravel() for a in (r, p, t, l, d, e, obj))

from biosppy.signals import ecg
from hrvanalysis import get_time_domain_features




def get_nn(peaks, fs):
    """Convert beat peaks in samples to NN intervals and timestamp."""
    rr = np.diff(peaks, prepend=0) * 1000 / fs

    # This remove outliers from signal
    rr = remove_outliers(rr, low_rri=300, high_rri=2000, verbose=False)
    # This replace outliers nan values with linear interpolation
    rr = interpolate_nan_values(rr, interpolation_method="linear")

    # This remove ectopic beats from signal
    # TODO: esto puede no tener sentido en PPG, pero los metodos de features
    #  estan basados en NN y no en RR.
    rr = remove_ectopic_beats(rr, method="malik", verbose=False)
    # This replace ectopic beats nan values with linear interpolation
    rr = np.array(interpolate_nan_values(rr))

    rr[np.where(np.isnan(rr))] = 0
    
    return np.array(rr)

from biosppy.signals import ecg

def process_ecg(signal, fs):
    """Get NN interval from ecg signal."""
    _, _, rpeaks, _, _, _, _ = ecg.ecg(signal,
                                       sampling_rate=fs,
                                       show=False)
    rr = get_nn(rpeaks, fs)

    return rr, rpeaks


def moving_hrv(rr, window_size, step_size, fs):
    window_samples = int(window_size * fs)
    step_samples = int(step_size * fs)
    hrv = []
    for i in range(0, len(rr) - window_samples, step_samples):
        rr_window = rr[i:i + window_samples]
        features = get_time_domain_features(rr_window)
        hrv.append(features['rmssd'])
    return hrv

def preprocess_physiology(data):
    
    index = data.index
    fs = 1000
    window_size = 10  # in seconds
    step_size = 1 / fs
    
    ecg_cleaned = nk.ecg_clean(data["ecg"])
    ecg_signals = pd.DataFrame({"ecg_cleaned": ecg_cleaned}, index=index)
    
    rr, peaks = process_ecg(data.ecg, fs)

    # Interpolate RR
    interpf = interpolate.interp1d(peaks/fs, rr, bounds_error=False,
                                fill_value=(rr[0], rr[-1]))

    # Create a time vector with the same length as the index
    timestamp = np.linspace(0, len(data["ecg"]) / fs, len(index))

    # Apply the interpolating function to the time vector
    rr_signal = interpf(timestamp)

    rr_signals = pd.DataFrame({"rr_signal": rr_signal}, index=index)
        
    #hrv = moving_hrv(rr_signal, window_size, step_size, fs)
    #hrv_df = pd.DataFrame(hrv, columns=["hrv"], index=index[:len(hrv)])

    # Preprocess BVP signal
    bvp_cleaned = nk.ppg_clean(data["bvp"])
    bvp_signals = pd.DataFrame({"bvp_cleaned": bvp_cleaned}, index=index)

    # Preprocess GSR signal
    gsr_cleaned = nk.eda_clean(data["gsr"], method='BioSPPy')
    gsr_signals = pd.DataFrame({"gsr_cleaned": gsr_cleaned}, index=index)

    # Calculate phasic and tonic components of GSR signal
    gsr_components = nk.eda_phasic(gsr_cleaned, sampling_rate=1000, method='cvxEDA')
    gsr_components.index = index
    gsr_components.columns = ['gsr_tonic', 'gsr_phasic']
    
    # Extrae Phasic, Tonic y SMNA activity de la señal GSR
    gsr_cleaned_crop = np.array_split(gsr_cleaned, 20)
    cvx = np.empty([3,0])
    for i in range(len(gsr_cleaned_crop)):
        [phasic_gsr, p, tonic_gsr, _ , _ , _ , _] = cvxEDA_pyEDA(gsr_cleaned_crop[i], 1./1000)
        if i == len(gsr_cleaned_crop)-1:
            phasic_gsr = np.zeros(len(gsr_cleaned_crop[i]))  # llenar con ceros
            tonic_gsr = np.full(len(gsr_cleaned_crop[i]), cvx_aux[2][-1])  # llenar con el último valor del fragmento anterior
        cvx_aux = np.vstack((phasic_gsr, p, tonic_gsr))
        cvx = np.hstack((cvx, cvx_aux))

    eda_phasic = cvx[0]
    eda_smna = cvx[1]
    eda_tonic = cvx[2]

    # Guarda la actividad del nervio sudomotor en gsr_SMNA
    gsr_SMNA = eda_smna

    # Agrega gsr_SMNA a la variable data
    gsr_SMNA = pd.DataFrame(gsr_SMNA, columns=["gsr_SMNA"], index=index)

    # Preprocess RSP signal
    rsp_cleaned = nk.rsp_clean(data["rsp"])
    rsp_signals = pd.DataFrame({"rsp_cleaned": rsp_cleaned}, index=index)
    fs =1000
    
    df_peaks, peaks_dict = nk.rsp_peaks(rsp_cleaned, sampling_rate=fs)
    info = nk.rsp_fixpeaks(peaks_dict)
    formatted = nk.signal_formatpeaks(info, desired_length=len(rsp_cleaned),peak_indices=info["RSP_Peaks"])
    formatted['RSP_Clean'] = rsp_cleaned
    
    # Extract rate
    rsp_rate = nk.rsp_rate(formatted, sampling_rate=fs)
    #rrv = nk.rsp_rrv(rsp_rate, info, sampling_rate=fs)
    rsp_rate_df = pd.DataFrame(rsp_rate, columns=["resp_rate"], index=index)

    # Preprocess EMG ZYGO signal
    emg_zygo_cleaned = nk.emg_clean(data["emg_zygo"])        
    emg_zygo_signals = pd.DataFrame({"emg_zygo_cleaned": emg_zygo_cleaned}, index=index)

    # Preprocess EMG CORU signal
    emg_coru_cleaned = nk.emg_clean(data["emg_coru"])
    emg_coru_signals = pd.DataFrame({"emg_coru_cleaned": emg_coru_cleaned}, index=index)

    # Preprocess EMG TRAP signal
    emg_trap_cleaned = nk.emg_clean(data["emg_trap"])
    emg_trap_signals = pd.DataFrame({"emg_trap_cleaned": emg_trap_cleaned}, index=index)

    # Preprocess Skin Temperature signal
    def butter_lowpass(cutoff, fs, order=5):
        return butter(order, cutoff, fs=fs, btype='low', analog=False)

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    # Filter requirements.
    order = 6
    fs = 1000.0     # sample rate, Hz (change this to match the sample rate of your skin temperature data)
    cutoff = 1      # desired cutoff frequency of the filter, Hz
   
    #SKT_filtered = low_pass_filter(data["skt"], cutoff_frequency=1.0, sampling_rate=1000)
    #skt_signals = pd.DataFrame({"skt_filtered": low_pass_filter(data["skt"], 1., 1000)}, index=index)
    #skt_signal = butter_lowpass_filter(data["skt"], cutoff, fs, order)
    #skt_signals = pd.DataFrame({"skt_filtered": skt_signal}, index=index)
    skt_signals = pd.DataFrame({"skt_filtered": data["skt"]}, index=index)

    # Combine preprocessed signals into one DataFrame
    preprocessed_data = pd.concat([ecg_signals,
                                    rr_signals,
#                                   hrv_df,
                                    bvp_signals,
                                    gsr_signals,
                                    gsr_components,
                                    gsr_SMNA,
                                    rsp_signals,
                                    rsp_rate_df,
                                    emg_zygo_signals,
                                    emg_coru_signals,
                                    emg_trap_signals,
                                    skt_signals], axis=1)

    return preprocessed_data

#%%
data = preprocess_physiology(train_physiology)


#%%
# Create a new figure
fig, ax = plt.subplots(14, 1, figsize=(15, 20), sharex=True)

# Plot all 8 physiological signals
for idx, signal in enumerate(data.columns):
    ax[idx].plot(data[signal], label=signal)
    ax[idx].set_ylabel(signal)
    ax[idx].legend()

# Plot the annotations timeseries
ax[13].plot(train_annotations, label="Annotations")
ax[13].set_ylabel("Annotations")
ax[13].legend()

# Set the xlabel for the last subplot
ax[13].set_xlabel("Time")

# Show the plot
plt.show()
#%%
# To-do
# [ ] Aplicar filtro pasa banda de SKT (por ahora quedó en raw esta señal, porque no pude aplicar los filtros)
# [x] Generar EDA_phasic, EDA_tonic, y EDA_SMNA a partir de data[gsr]
# [x] Generar EDA_SMNA a partir de data[gsr]
# [x] Implementar obtencion de RR peak (medida continua) y HRV (variable continua) a partir de data de ECG
# [x] Obtener RRV de respiration rate (variabilidad de respiracion, como variable continua)

#%%
import pandas as pd

file_path = '../data/preprocessed/cleaned_and_prepro/scenario_1/test/physiology/sub_11_vid_1.csv'

data = pd.read_csv(file_path)

#%%
# Create a new figure
fig, ax = plt.subplots(14, 1, figsize=(15, 20), sharex=True)

# Plot all 8 physiological signals
for idx, signal in enumerate(data.columns):
    ax[idx].plot(data[signal], label=signal)
    ax[idx].set_ylabel(signal)
    ax[idx].legend()

# Plot the annotations timeseries
ax[13].plot(train_annotations, label="Annotations")
ax[13].set_ylabel("Annotations")
ax[13].legend()

# Set the xlabel for the last subplot
ax[13].set_xlabel("Time")

# Show the plot
plt.show()


# %%
