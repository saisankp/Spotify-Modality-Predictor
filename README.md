# Spotify Modality Predictor

<img align="right" src="https://cdn.dribbble.com/users/411475/screenshots/13544773/media/ea8bb04d924af2ada6c128ce8d234442.jpg?compress=1&resize=1600x1200&vertical=top" width="300" height="220">
<p align=justify>
 This is group 72's project for the module <strong>Machine Learning (CSU44061)</strong> at Trinity College Dublin. As a musician, the modality of a song (whether a song is written in a major key or minor key) is extremely important. The modality of a song tells us a lot about it - for example, songs written in a major key are identified as happy meanwhile songs written in a minor key are identified as sad in western culture. Our project aims to predict the modality of songs on Spotify. Using a song's audio features, you can use machine learning models to predict the modality of a song. The prediction of modality an be used for various purposes such as helping streamline the process of uploading music for Spotify artists (instead of them manually inputting this information), improving Spotify's song uploading process, and recommending songs based on modality predictions to users.</i>
</p>

## What machine learning models does this project use?

  1. kNN classifier (k-nearest neighbours).
  2. Kernalised SVM (support vector machine).
  
  
## What do I need before I run the project?

  1. Python (which you can download [here][PythonDownload]).
  2. Pip (which you can download [here][PipDownload]).
  3. PyCharm IDE (this is only a recommendation, but it seems to work well with this project - you can install it [here][PyCharmDownload]).
  3. A Spotify API key (if you wish to do data collection, you can get one from [here][SpotifyAPIKey]).

## How can I get started running this project locally?
  1. To setup the .env file, run `bash setupEnv.sh` and input your Spotify API key (only necessary if you want to use the file data_collection.py). This will make a .env file inside of the src directory.
  2. If you use PyCharm, it will help you download all necessary libraries such as scikit-learn. However, the progress bar library might struggle to install, in which case you can simply run `pip install progressbar2` to install it.
  3. To run any python file, you can use PyCharm or simply run `python3 main.py` for example to execute it.

## Group 72 Members

|    Student Name    | Student ID |                Course                |      GitHub Username       |
|:------------------:|:----------:|:------------------------------------:|:--------------------------:|
|   Prathamesh Sai   |  19314123  | Integrated Computer Science (M.C.S.) |    [saisankp][saisankp]    |
| Eligijus Skersonas |  19335661  | Integrated Computer Science (M.C.S.) | [eli-scorpio][eli-scorpio] |


[saisankp]: https://github.com/saisankp
[eli-scorpio]: https://github.com/eli-scorpio
[PythonDownload]: https://www.python.org/downloads/
[PipDownload]: https://pypi.org/project/pip/
[PyCharmDownload]: https://www.jetbrains.com/pycharm/download/
[SpotifyAPIKey]: https://developer.spotify.com/console/
