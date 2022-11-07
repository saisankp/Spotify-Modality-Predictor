import os
import time
import progressbar
import requests
from dotenv import load_dotenv
import csv


# Function getListOfSongIDFromGenre returns a list of 50 song ID's for a particular genre.
# It takes in the offset (index of the search results to start from), and the requested genre.
# It returns an array of size 50 containing track ID's representing tracks of the requested genre.
# It works by using the 'search by genre' functionality of the Spotify API. You can use the returned ID's at
# https://open.spotify.com/track/{Enter-ID-Here-Without-Brackets} to listen to a particular song.
def getListOfSongIDFromGenre(offset, genre):
    listOfSongID = []
    spotifySearchURL = "https://api.spotify.com/v1/search"
    params = {
        "q": "genre:" + genre,
        "type": "track",
        "limit": "50",
        "offset": offset
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.getenv("SPOTIFY_API_TOKEN"),
    }
    songs = requests.get(spotifySearchURL, params=params, headers=headers)
    songs = songs.json()
    if ("Invalid access token" in str(songs) or "The access token expired" in str(songs) or "Invalid limit" in str(
            songs)):
        raise Exception(songs)
    if "API rate limit exceeded" in str(songs):
        time.sleep(5)
        return getListOfSongIDFromGenre(offset, genre)
    else:
        for track in range(50):
            listOfSongID.append(songs["tracks"]["items"][track]["id"])
        return listOfSongID


# Function getListOfAudioFeatures returns a list of lists, where each index represents the audio features of a song.
# It takes in a list of song ID's where the user wants to find their audio features. Each index looks like:
# [Spotify Song URL, danceability, energy, valence, sentiment]
# This is calculated by using the 'audio-features' functionality of the Spotify API. The function returns -1 in error
# cases (If there is no data available for the song or the input list is empty).
def getListOfAudioFeatures(listOfSongID, sentiment):
    if listOfSongID == [None] or listOfSongID == -1:
        return -1
    else:
        listOfAudioFeatures = []
        for indexOfSongID in range(len(listOfSongID)):
            songFeatures = []
            spotifyGetAudioFeaturesURL = "https://api.spotify.com/v1/audio-features"
            params = {
                "ids": str(listOfSongID[indexOfSongID])
            }
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": "Bearer " + os.getenv("SPOTIFY_API_TOKEN")
            }
            audioFeaturesForSong = requests.get(spotifyGetAudioFeaturesURL, params=params, headers=headers)
            audioFeaturesForSong = audioFeaturesForSong.json()
            # If the API call had an error (time-out limit reached/no response), we can retry with same parameters.
            if not ("audio_features" in str(audioFeaturesForSong)):
                return getListOfAudioFeatures(listOfSongID, sentiment)
            elif audioFeaturesForSong == [None]:
                return getListOfAudioFeatures(listOfSongID, sentiment)
            elif audioFeaturesForSong["audio_features"] == [None]:
                return getListOfAudioFeatures(listOfSongID, sentiment)
            else:
                songFeatures.append("https://open.spotify.com/track/" + str(listOfSongID[indexOfSongID]))
                songFeatures.append(audioFeaturesForSong["audio_features"][0]["danceability"])
                songFeatures.append(audioFeaturesForSong["audio_features"][0]["energy"])
                songFeatures.append(audioFeaturesForSong["audio_features"][0]["valence"])
                songFeatures.append(sentiment)
                listOfAudioFeatures.append(songFeatures)
        return listOfAudioFeatures


# Function writeFeaturesToFile takes in features from 2 steps, and writes them into the requested CSV file.
def writeFeaturesToFile(featuresFromStep1, featuresFromStep2, filename):
    with open(filename, "w", newline="") as file:
        header = ["Spotify-Song", "X1", "X2", "X3", "y"]
        writer = csv.writer(file)
        writer.writerow(header)
        for dataFromStepNMinus1 in range(len(featuresFromStep1)):
            for Step1 in range(len(featuresFromStep1[dataFromStepNMinus1])):
                writer.writerow(featuresFromStep1[dataFromStepNMinus1][Step1])
        for dataFromStepNMinus2 in range(len(featuresFromStep2)):
            for Step2 in range(len(featuresFromStep2[dataFromStepNMinus2])):
                writer.writerow(featuresFromStep2[dataFromStepNMinus2][Step2])


if __name__ == '__main__':
    # INTRO: For data collection, we want to get a wide range of songs with a wide range of sentiment level for the best
    # results. Firstly, we want to use the Spotify API to search across Spotify for happy songs within a "happy" genre.
    # The songs from this search are guaranteed to have varying levels of positive sentiment. Secondly, we want to
    # use the Spotify API to search across Spotify for sad songs within a "sad" genre. The songs from this search are
    # guaranteed to have varying levels of negative sentiment. Both of these searches will allow the data to be diverse.

    # Load the Spotify API Key from the .env file.
    # (You can get a key from https://developer.spotify.com/console/get-search-item/ by pressing "Get Token")
    load_dotenv()

    # To adhere to the Spotify API limit, we need to break up the API calls in segments of 50 songs per API call.
    # We can therefore use a list to append the features of each iteration (hence features of every 50 songs).
    featuresFromPositiveSentiment = []
    featuresFromNegativeSentiment = []

    # STEP 1. Add 1000 songs with varying levels of positive sentiment:
    print("Step 1: Add 1000 songs with positive sentiment to the data set. (1/3)")
    time.sleep(0.05)  # Ensure print statement is printed before progress bar.

    # Since Spotify has a limit of 50 songs per API call, we need to increase the offset by 50 for every API call,
    # so that we get the next 50 songs for every API call from the Spotify search.
    offset = 0

    for i in progressbar.progressbar(range(20)):
        # Search Spotify to get a list of 50 song ID's from the "happy" genre.
        listOfSongID = getListOfSongIDFromGenre(offset, "happy")
        # Get the audio features of every song ID, passing in the expected sentiment (1 = positive).
        features = getListOfAudioFeatures(listOfSongID, 1)
        featuresFromPositiveSentiment.append(features)
        offset = offset + 50

    # STEP 2. Add 1000 songs with varying levels of negative sentiment:
    print("\n" + "Step 2: Add 1000 songs with negative sentiment to the data set. (2/3)")
    time.sleep(0.05)  # Ensure print statement is printed before progress bar.

    # Since Spotify has a limit of 50 songs per API call, we need to increase the offset by 50 for every API call,
    # so that we get the next 50 songs for every API call from the Spotify search.
    offset = 0

    for i in progressbar.progressbar(range(20)):
        # Search Spotify to get a list of 50 song ID's from the "sad" genre.
        listOfSongID = getListOfSongIDFromGenre(offset, "sad")
        # Get the audio features of every song ID, passing in the expected sentiment (0 = negative).
        features = getListOfAudioFeatures(listOfSongID, 0)
        featuresFromNegativeSentiment.append(features)
        offset = offset + 50

    # STEP 3. Write the collected data into the dataset.csv file:
    print("\n" + "Step 3: Write 2000 songs with positive and negative sentiment to dataset.csv. (3/3)")
    writeFeaturesToFile(featuresFromPositiveSentiment, featuresFromNegativeSentiment, "dataset.csv")
