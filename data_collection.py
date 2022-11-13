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
    elif ("API rate limit exceeded" in str(songs)) or (not songs.items):
        time.sleep(5)
        return getListOfSongIDFromGenre(offset, genre)
    elif len(songs["tracks"]["items"]) == 0:
        return -1
    else:
        for track in range(50):
            listOfSongID.append(songs["tracks"]["items"][track]["id"])
        return listOfSongID


# Function getListOfAudioFeatures returns a list of lists, where each index represents the audio features of a song.
# It takes in a list of song ID's where the user wants to find their audio features. Each index looks like:
# [Spotify Song URL, danceability, energy, valence, sentiment]
# This is calculated by using the 'audio-features' functionality of the Spotify API. The function returns -1 in error
# cases (If there is no data available for the song or the input list is empty).
def getListOfAudioFeatures(listOfSongID):
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
                return getListOfAudioFeatures(listOfSongID)
            elif audioFeaturesForSong == [None]:
                return getListOfAudioFeatures(listOfSongID)
            elif audioFeaturesForSong["audio_features"] == [None]:
                return getListOfAudioFeatures(listOfSongID)
            else:
                songFeatures.append("https://open.spotify.com/track/" + str(listOfSongID[indexOfSongID]))
                songFeatures.append(audioFeaturesForSong["audio_features"][0]["danceability"])
                songFeatures.append(audioFeaturesForSong["audio_features"][0]["energy"])
                songFeatures.append(audioFeaturesForSong["audio_features"][0]["key"])
                songFeatures.append(audioFeaturesForSong["audio_features"][0]["loudness"])
                songFeatures.append(audioFeaturesForSong["audio_features"][0]["speechiness"])
                songFeatures.append(audioFeaturesForSong["audio_features"][0]["acousticness"])
                songFeatures.append(audioFeaturesForSong["audio_features"][0]["instrumentalness"])
                songFeatures.append(audioFeaturesForSong["audio_features"][0]["liveness"])
                songFeatures.append(audioFeaturesForSong["audio_features"][0]["tempo"])
                songFeatures.append(audioFeaturesForSong["audio_features"][0]["valence"])
                songFeatures.append(audioFeaturesForSong["audio_features"][0]["duration_ms"])
                songFeatures.append(audioFeaturesForSong["audio_features"][0]["time_signature"])
                if audioFeaturesForSong["audio_features"][0]["mode"] == 0:
                    songFeatures.append(-1)
                elif audioFeaturesForSong["audio_features"][0]["mode"] == 1:
                    songFeatures.append(1)
                listOfAudioFeatures.append(songFeatures)
        return listOfAudioFeatures


# Function writeFeaturesToFile takes in features from 2 steps, and writes them into the requested CSV file.
def writeFeaturesToFile(featuresFromStep1, filename):
    with open(filename, "w", newline="") as file:
        header = ["Spotify-Song", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "y"]
        writer = csv.writer(file)
        writer.writerow(header)
        for dataFromStepNMinus1 in range(len(featuresFromStep1)):
            for Step1 in range(len(featuresFromStep1[dataFromStepNMinus1])):
                writer.writerow(featuresFromStep1[dataFromStepNMinus1][Step1])


if __name__ == '__main__':
    # Load the Spotify API Key from the .env file.
    # (You can get a key from https://developer.spotify.com/console/get-search-item/ by pressing "Get Token")
    load_dotenv()

    # Every genre on spotify (from https://developer.spotify.com/console/get-available-genre-seeds/)
    genres = ["acoustic", "afrobeat", "alt-rock", "alternative", "ambient", "anime", "black-metal", "bluegrass",
              "blues", "brazil", "breakbeat", "british", "cantopop", "chicago-house", "children",
              "chill", "classical", "club", "comedy", "country", "dance", "dancehall", "death-metal", "deep-house",
              "detroit-techno", "disco", "disney", "drum-and-bass", "dub", "dubstep", "edm", "electro", "electronic",
              "emo", "folk", "forro", "french", "funk", "garage", "german", "gospel", "goth", "grindcore", "groove",
              "grunge", "guitar", "happy", "hard-rock", "hardcore", "hardstyle", "heavy-metal", "hip-hop",
              "honky-tonk", "house", "idm", "indian", "indie", "indie-pop", "industrial", "iranian", "j-dance",
              "j-idol", "j-pop", "j-rock", "jazz", "k-pop", "kids", "latin", "latino", "malay", "mandopop", "metal",
              "metalcore", "minimal-techno", "mpb", "new-age", "opera", "pagode",
              "party", "piano", "pop", "pop-film", "power-pop", "progressive-house",
              "psych-rock", "punk", "punk-rock", "r-n-b", "reggae", "reggaeton", "rock",
              "rock-n-roll", "rockabilly", "romance", "sad", "salsa", "samba", "sertanejo", "show-tunes",
              "singer-songwriter", "ska", "sleep", "songwriter", "soul",  "spanish", "study",
              "swedish", "synth-pop", "tango", "techno", "trance", "trip-hop", "turkish", "world-music"]
    featuresFromEveryGenre = []
    for i in progressbar.progressbar(range(len(genres))):
        # Get 50 songs ID's of songs from one genre
        listOfSongID = getListOfSongIDFromGenre(0, genres[i])
        # Get the audio features of every song ID
        features = getListOfAudioFeatures(listOfSongID)
        # Add this list audio features to the overall list of features
        featuresFromEveryGenre.append(features)

    writeFeaturesToFile(featuresFromEveryGenre, "dataset.csv")