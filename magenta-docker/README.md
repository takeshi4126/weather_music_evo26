
# Docker container to run MelodyRNN

This folder contains the Dockerfile and some scripts to run the MelodyRNN [1] to generate weather music.

## What is MelodyRNN

MelodyRNN is a pre-trained music generation model based on the LSTM. It is a part of the Google Magenta project.

Github repo: https://github.com/magenta/magenta

## Why it is needed

If you're using an Intel-based architecture, you may be able to install Magenta natively, following the Installation instruction at the Github repo (or any other instructions available on the Internet.)

However, if you're using an ARM-based computer, like my Macbook, and if you cannot install Mangeta directly there, then the docker file in this folder may help you.

## How it is used

1. Build the docker image.
```
docker build -t magenta-minimal .
```

2. Run the start_docker.sh script and confirm that the docker container named "my-magenta-container" is up and running.

WeatherSonification.py under the models folder executes the MelodyRNN in the started "my-magenta-container" docker container.


