# Weather Sonification via a Latent Emotion Space

This repository contains the script to convert weather data into music (sonification).

## Input data

The following data are used to train the models.

1. Numerical Weather Prediction (NWP) data from the UK Met Office [1]. The downloaded and converted data for year 2023 is found as data/input_data/asdi_data.parquet.
2. MIDI files from https://github.com/zBaymax/EIMG provided by the work [2]. It should be extracted under data/input_data/dataset folder. e.g. data/input_data/dataset/0001.mid
3. The Excel file containing the annotation of valence & arousal values to each MIDI files, provided by the work [2]. Download it from the same repository above and put it in the data/input_data folder.
4. The Excel file containing the annotation of valence, arousal, dominance and surprise values, provided by the work [3]. Download it from https://www.mdpi.com/2073-4433/11/8/860/s1 and put the "Aﬀective Normative Data for English Weather Words supp Table S1.xlsx" file in the data/input_data folder.

## Execution

|#|File|Contents|
|---|---|---|
|1|0.1-Exploratory_Data_Analysis.ipynb|EDA of the NWP data from the UK Met Office rolling archive.|
|2|1.1-WeatherVAE.ipynb|Run the WeatherVAE to capture weather embeddings in the latent space.|
|3|1.2-WeatherVAEValidation.ipynb|5-fold validation of the WeatherVAE.|
|4|2.1-ExtractMusicFeatures.ipynb|EDA of the music data to identify important music features.|
|5|2.2-MusicEmotionVAE.ipynb|Train the Music-Emotion VAE to obtain the latent emotion space.|
|6|3.1-WeatherEmotionFNN.ipynb|Train the Weather-Emotion FNN to obtain the latent emotion space.|
|7|4.1-WeatherMusicAssociation.ipynb|Run the Weather-Music Association ML model to infer the music features for the input weather data.|
|8|5.1-WeatherSonificationEvaluation.ipynb|Generate the music files for the listening evaluation.|
|9|5.2-WeatherSonificationEvaluationAnalysis.ipynb|Supporting visualisation of the generated music samples for evaluation.|

The notebooks at step 8 uses the MelodyRNN as the LSTM to generate the MIDI files. The MelodyRNN must be run separately before running the notebooks. Since MelodyRNN does not run in the author's computer (Mac OS with M1 chipset) natively, it was run as a docker container. See the magenta-docker folder at the top directory. WeatherSonification.py file contains the code to execute the MelodyRNN in the docker environment.

## Output

MIDI files will be generated in the midi_for_evaluation_2 folder. samples.csv file there contains the weather data and the date & time used for generating each MIDI file. You can open the midi files by using MuseScore (https://musescore.org).  

## Output used for the Listening Evaluation

The output used for the Listening Evaluation in the paper is found under data/midi_for_evaluation. The "samples.csv" file in the folder contains the weather parameters and music features used for the sonification of the 20 music files. data/mp3_for_evaluation folder contains the sample music files but in mp3 format for the listeners who cannot play the midi files.

## Folders

- src folder contains the source code files including the Jupyter notebooks.
- src/model folder contains the python source code for the ML models. They are imported by one or more Jupyter notebook files listed above.
- src/common folder contains some functions shared by multiple Jupyter notebook and model scripts.
- src/optuna folder contains the scripts to run the model with the Optuna hyper-parameter search framework [4].

## References

[1] UK Met Office: Met Office UK Deterministic (UKV) 2km on a 2-year rolling archive, https://registry.opendata.aws/met-office-uk-deterministic/, last ac-cessed 2025/08/31.

[2] Y. Wang, M. Chen and X. Li, "Continuous Emotion-Based Image-to-Music Generation," IEEE Transactions on Multimedia, vol. 26, pp. 5670-5679, 2024. 

[3] A. E. Stewart, "Affective Normative Data for English Weather Words," Atmosphere, vol. 11, no. 8, p. 860, 2020. 

[4] Akiba, T., Sano, S., Yanase, T., Ohta, T., Koyama, M.: Optuna: A Next-generation Hyperparameter Optimization Framework. In: KDD ’19: Proceed-ings of the 25th ACM SIGKDD International Conference on Knowledge Dis-covery & Data Mining (2019). https://doi.org/10.1145/3292500.3330701.

