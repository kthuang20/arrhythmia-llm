### import necessary packages
import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk

class ECGProcessor():
    def __init__(self, data_path, patient_id, segment_id):
        """Initializes attributes of ECG signal analysis to store and loads in an ECG signal."""
        self.data_path = data_path
        self.patient_id = patient_id
        self.segment_id = segment_id

        self.ecg = None # will store cleaned ECG
        self.fs = None # will store sampling frequency
        self.ann = None
        self.rpeaks = None # will store the locations of each peak
        self.labels = None # will store only the beat annotations
        self.beat_segments = None # will store heartbeat signals
        self.beat_analysis = None # will store analysis of ECG features across all heartbeats
        
        # load ECG and annotations
        self.ecg, self.ann, self.rpeaks, self.labels, self.fs = self.load_ecg()
        
    def load_ecg(self):
        """Loads the entire ECG segment of a patient and filters annotations for classifiable heartbeats."""
        # loads in ECG signal
        filename = f'{self.data_path}/p0{str(self.patient_id)[:1]}/p{self.patient_id:05d}/p{self.patient_id:05d}_s{self.segment_id:02d}'
        rec = wfdb.rdrecord(filename)
        ecg_raw = rec.p_signal[:, 0]
        ann = wfdb.rdann(filename, "atr")
        fs = rec.fs # sampling rate
        rpeaks = ann.sample # location of peak
        labels = ann.symbol # beat annotations

        # clean ECG signal (passes through a 0.5 Hz high-pass butterworth filter)
        ecg_clean, info = nk.ecg_process(ecg_raw, sampling_rate=fs)

        return ecg_clean, ann, rpeaks, labels, fs

    def segment_by_beats(self):
        """Segments ECG signal by heartbeats."""
        # convert beat annotations to numpy array
        labels_array = np.array(self.labels)
        # create a filter to select for only beat annotations 
        beat_labels = (labels_array != "Q") & (labels_array != "+")
        # update beat annotations
        self.rpeaks = self.rpeaks[beat_labels]
        self.labels = labels_array[beat_labels].tolist()
        
        # separate the ECG by heartbeats
        beats = nk.epochs_create(
            data=self.ecg,
            events=self.rpeaks,
            sampling_rate=self.fs,
            epochs_start=-0.2,
            epochs_end=0.4
        )

        # store heartbeats
        self.beat_segments = beats
        
        return beats

    def analyze_beats(self):
        """Analyze the ECG features across entire ECG signal."""
        # analyze features across entire ECG signal
        beat_analysis = nk.ecg_analyze(self.ecg, sampling_rate=self.fs)

        # keep only relevant clinical features
        features = ["ECG_Rate_Mean", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50", "HRV_SD1", "HRV_SampEn"]
        beat_analysis = beat_analysis[features]
        
        # # analyze the features across all heartbeats
        # beat_analysis = nk.ecg_analyze(self.beat_segments, sampling_rate=self.fs)
        # # keep only relevant clinical features from results
        # features2remove = ["Label", "Event_Onset", "ECG_Rate_Baseline", "ECG_Rate_Trend_Linear", "ECG_Rate_Trend_Quadratic",
        #                    "ECG_Rate_Trend_R2", "ECG_Quality_Mean"]
        # beat_analysis = beat_analysis.loc[:, ~beat_analysis.columns.isin(features2remove)]

        # store results from analysis
        self.beat_analysis = beat_analysis
        
        return beat_analysis
