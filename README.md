# level-crossing-alarm-classification
CNN to identify level crossing alarm bell among different types of alarms


# Infos

Level Crossings are among the most critical railway assets, concerning both the risk of accidents and their maintainability, due to intersections with promiscuous traffic and difficulties in remotely monitoring their health status. Failures can be originated from several factors, including malfunctions in the bar mechanisms and warning devices, such as light signals and bells. 

# Contest Rules

Each student has to predict (multi-class categorical classification) the source of the sound provided as a sample, realising one or more prediction models using data analysis and Machine Learning techniques. The performance measure to maximise is Accuracy. It is mandatory for the student who will achieve the best performance on the test dataset, to discuss the process steps followed in order to reach the development of the final model. The winning student presentation will be held during the lesson on December the 17th.
If the presentation and the proposed solution will be judged positively, the author will be relieved from discussing one of the contest solutions during the final exam.
Each participant is free to use external tools (i.e. Weka, Knime, MatLab, etc.).

# Specs

This contest uses audio data samples, represented by their MEL spectrogram. Each sample is memorised into a NumPy array having three columns:
The mel-spectogram data of the i-th sample. It is a (96*#sec) x 64 elements matrix, where each 96x64 sub-matrix corresponds to 1sec in the original audio (e.g. a 10 seconds audio will results in a 960 x 64 element matrix);
The categorical class, codified as follows: Warning bell -> 0; No Alarm -> 1, Generic Alarm -> 2;
The one-hot encoded class, codified as follows: Warning bell -> 1 0 0, No Alarm -> 0 1 0, -> Generic Alarm -> 0 0 1;
The name of the file corresponds to the sample's id. The samples are gathered into three zip files (training, test, validation). Participants are free to use any type of feature extraction approach (e.g. using a pre-trained CNN, LBP, GLCM, etc.) or train an ad-hoc model. Samples in the test set are completely separated from the training one.
