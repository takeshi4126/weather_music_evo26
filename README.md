# Weather Sonification via a Latent Emotion Space

## Input data

The following data are used to train the models.

1. MIDI files from https://github.com/zBaymax/EIMG provided by the work [1]. There is dataset.zip file in the repository. It should be extracted under input_data/dataset folder. e.g. input_data/dataset/0001.mid
2. The Excel file containing the annotation of valence & arousal values to each MIDI files, provided by the work [1]. Download it from the same repository above and put it in this folder.
3. The Excel file containing the annotation of valence, arousal, dominance and surprise values, provided by the work [2]. Download it from https://www.mdpi.com/2073-4433/11/8/860/s1 and put the "Aﬀective Normative Data for English Weather Words supp Table S1.xlsx" file in this folder.

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

## Evaluation

midi_for_evaluation folder contains the MIDI files used for the listening evaluation (see Section 6.2.6 of the dissertation report). samples.csv file contains the weather data and the date & time used for generating each MIDI file. Listening evaluation was conducted without reading the sample.csv file. 

## Folders

- model folder contains the python source code for the ML models. They are imported by one or more Jupyter notebook files listed above.

## References

[1] Y. Wang, M. Chen and X. Li, "Continuous Emotion-Based Image-to-Music Generation," IEEE Transactions on Multimedia, vol. 26, pp. 5670-5679, 2024. 
[2] A. E. Stewart, "Affective Normative Data for English Weather Words," Atmosphere, vol. 11, no. 8, p. 860, 2020. 
