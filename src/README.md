
SEGUIR DESDE ACA ->
MEJORAR EL ENTENDIMIENTO DE LAS SEÑALES EXTRAIDAS DE `PREPROCESS.PY`.
DEPURAR EL PROCESO DE EXTRACCION DE SEÑALES BUSCANDO PAPERS AL RESPECTO DE FEATURE EXTRACCION DE SEÑALES.
¿ES NECESARIO ESTO TAMBIÉN CUANDO HAGO DEEP LEARNING? EN ESTE CASO TAL VEZ SOLO SEA NECESARIO LIMPIAR LAS SEÑALES.
PUEDO GUARDAR UNA VERSION CON DATA CLEAN, Y OTRA VERSION CON DATA DE FEATURES EXTREACTED.

PASOS A SEGUIR:
1. ENTENDER QUE SON CADA UNA DE LAS SEÑALES EXTRAIDAS EN `PREPROCESS.PY`
2. BUSCAR DATOS DE FEATURES REALIZACIONADAS A CADA SEÑAL FISIOLOGICA (BUSCAR EN PAPERS QUE USARON ESTA BASE DE DATOS EN PARTICULAR) Y AGREGAR COMO FEATURES A EXTRAER EN PREPROCESS.
3. GUARDAR DATOS CLEANED EN `./PREPROCESSED` Y TAMBIEN DATOS CON FEATURES EN EL MISMO DIRECOTRIO
4. CORRER MODELO LINEA DE BASE CON AMBOS TIPOS DE DATOS EN ARETE (HABIENDO ELEGIDO PRIMERO UNA VENTANA DE INTERÉS APROPIADA,SEGUN REVISION DE LA LITERATURA)


# Preprocessed Physiological Data

This README file explains the output variables of the `preprocess_physiology` function.

## Output Variables

The output DataFrame of the `preprocess_physiology` function contains the following columns:

### Electrocardiogram (ECG) Signal
- ECG_Raw: Raw ECG signal
- ECG_Clean: Cleaned ECG signal
- ECG_Rate: Instantaneous heart rate derived from the ECG signal
- ECG_Quality: Signal quality index
- ECG_R_Peaks: R-peaks in the ECG signal
- ECG_P_Peaks: P-peaks in the ECG signal
- ECG_P_Onsets: P-wave onsets
- ECG_P_Offsets: P-wave offsets
- ECG_Q_Peaks: Q-peaks in the ECG signal
- ECG_R_Onsets: R-wave onsets
- ECG_R_Offsets: R-wave offsets
- ECG_S_Peaks: S-peaks in the ECG signal
- ECG_T_Peaks: T-peaks in the ECG signal
- ECG_T_Onsets: T-wave onsets
- ECG_T_Offsets: T-wave offsets
- ECG_Phase_Atrial: Atrial phase of the cardiac cycle
- ECG_Phase_Completion_Atrial: Atrial phase completion
- ECG_Phase_Ventricular: Ventricular phase of the cardiac cycle
- ECG_Phase_Completion_Ventricular: Ventricular phase completion

### Photoplethysmogram (PPG) Signal
- PPG_Raw: Raw PPG signal
- PPG_Clean: Cleaned PPG signal
- PPG_Rate: Instantaneous heart rate derived from the PPG signal
- PPG_Peaks: Peaks in the PPG signal

### Electrodermal Activity (EDA) Signal
- EDA_Raw: Raw EDA signal
- EDA_Clean: Cleaned EDA signal
- EDA_Tonic: Tonic component of the EDA signal
- EDA_Phasic: Phasic component of the EDA signal
- SCR_Onsets: Skin conductance response (SCR) onsets
- SCR_Peaks: SCR peaks
- SCR_Height: SCR height
- SCR_Amplitude: SCR amplitude
- SCR_RiseTime: SCR rise time
- SCR_Recovery: SCR recovery
- SCR_RecoveryTime: SCR recovery time

### Respiration (RSP) Signal
- RSP_Raw: Raw RSP signal
- RSP_Clean: Cleaned RSP signal
- RSP_Amplitude: Respiration amplitude
- RSP_Rate: Respiration rate
- RSP_RVT: Respiratory volume per time (RVT)
- RSP_Phase: Respiratory phase
- RSP_Phase_Completion: Respiratory phase completion
- RSP_Symmetry_PeakTrough: Peak-trough symmetry in respiration signal
- RSP_Symmetry_RiseDecay: Rise-decay symmetry in respiration signal
- RSP_Peaks: Respiration peaks
- RSP_Troughs: Respiration troughs

### Electromyography (EMG) Signal (for each EMG channel)
- EMG_Raw: Raw EMG signal
- EMG_Clean: Cleaned EMG signal
- EMG_Amplitude: EMG amplitude
- EMG_Activity: EMG activity
- EMG_Onsets: EMG onsets