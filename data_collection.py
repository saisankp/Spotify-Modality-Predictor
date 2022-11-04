import os
import requests
from dotenv import load_dotenv
import csv


# Function getListOfSongIDFromGenre takes in the number of tracks (n) and genre.
# It returns an array of size n containing track ID's representing tracks of the requested genre.
# It works by using the 'search by genre' functionality of the Spotify API. You can use the returned ID's at
# https://open.spotify.com/track/{Enter-ID-Here-Without-Brackets} to listen to the song if you wish.
def getListOfSongIDFromGenre(numberOfTracks, genre):
    listOfSongID = []
    spotifySearchURL = "https://api.spotify.com/v1/search"
    params = {
        "q": "genre:" + genre,
        "type": "track",
        "limit": str(numberOfTracks)
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
    else:
        for track in range(numberOfTracks):
            listOfSongID.append(songs["tracks"]["items"][track]["id"])
        return listOfSongID


# Function getListOfAudioFeatures takes in a list of song ID's and the sentiment (0 or 1). It returns an array,
# the same size as the input array containing an array of arrays, where each index contains an array filled with
# audio features for that particular song. Therefore, each element in the array (i.e. song) would contain:
# [Danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, tempo, valence,
# sentiment]. This is calculated by using the 'audio-features' functionality of the Spotify API. The function returns -1
# in error cases (If there is no data available for the song or the input list is empty).
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
            if not ("audio_features" in str(audioFeaturesForSong)):
                return -1;
            elif audioFeaturesForSong == [None]:
                return -1
            elif audioFeaturesForSong["audio_features"] == [None]:
                return -1
            else:
                songFeatures.append(audioFeaturesForSong["audio_features"][0]["danceability"])
                songFeatures.append(audioFeaturesForSong["audio_features"][0]["energy"])
                songFeatures.append(audioFeaturesForSong["audio_features"][0]["key"])
                songFeatures.append(audioFeaturesForSong["audio_features"][0]["loudness"])
                songFeatures.append(audioFeaturesForSong["audio_features"][0]["mode"])
                songFeatures.append(audioFeaturesForSong["audio_features"][0]["speechiness"])
                songFeatures.append(audioFeaturesForSong["audio_features"][0]["acousticness"])
                songFeatures.append(audioFeaturesForSong["audio_features"][0]["instrumentalness"])
                songFeatures.append(audioFeaturesForSong["audio_features"][0]["liveness"])
                songFeatures.append(audioFeaturesForSong["audio_features"][0]["tempo"])
                songFeatures.append(audioFeaturesForSong["audio_features"][0]["valence"])
                songFeatures.append(sentiment)
                listOfAudioFeatures.append(songFeatures)
        return listOfAudioFeatures


# Function getListOfSongIDFromGenreAndMood takes the number of tracks (n), the genre, and sentiment (a string
# representing the level of sentiment).  The function returns a list of song ID's representing songs with the
# requested genre and level of sentiment. This is calculated by using the 'search for playlist' functionality
# of the Spotify API which returns the playlist ID. Using the playlist ID, I get the first song in the playlist
# by using the 'get tracks from playlist' functionality from the Spotify API. The function returns -1 in error cases
# (If the song in the acquired playlist has been removed or the playlist itself has no songs).
def getListOfSongIDFromGenreAndMood(numberOfTracks, genre, sentiment):
    listOfSongID = []
    # Firstly, we try to get 1 public playlist named with the GENRE and SENTIMENT that we want.
    spotifyPlaylistSearchURL = "https://api.spotify.com/v1/search"
    params = {
        "q": genre + " " + sentiment,
        "type": "playlist",
        "limit": "1"
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.getenv("SPOTIFY_API_TOKEN"),
    }
    playlist = requests.get(spotifyPlaylistSearchURL, params=params, headers=headers)
    playlist = playlist.json()
    if ("Invalid access token" in str(playlist) or "The access token expired" in str(
            playlist) or "Invalid limit" in str(
        playlist)):
        raise Exception(playlist)

    elif not playlist["playlists"]["items"]:
        return -1;
    else:
        # Now we have 1 playlist ID of a playlist with the necessary GENRE and SENTIMENT.
        playListID = playlist["playlists"]["items"][0]["id"]
        # Secondly, we need to get N number of tracks from that playlist (as much as we need).
        spotifyGetPlayListTracks = "https://api.spotify.com/v1/playlists/" + playListID + "/tracks"
        params = {
            "limit": str(numberOfTracks)
        }
        songs = requests.get(spotifyGetPlayListTracks, params=params, headers=headers)
        songs = songs.json()
        if ("Invalid access token" in str(songs) or "The access token expired" in str(songs) or "Invalid limit" in str(
                songs)):
            raise Exception(songs)
        # Iterate through as many tracks as we need, and add it to a list.
        for track in range(numberOfTracks):
            if songs["items"][track]["track"] is None:
                return -1
            else:
                listOfSongID.append(songs["items"][track]["track"]["id"])
        return listOfSongID


# Function writeFeaturesToFile takes in features from 4 steps, and writes them into the requested CSV file.
def writeFeaturesToFile(featuresFromStep1, featuresFromStep2, featuresFromStep3, featuresFromStep4, filename):
    with open(filename, "w", newline="") as file:
        header = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "y"]
        writer = csv.writer(file)
        writer.writerow(header)
        for dataFromStepNMinus1 in range(len(featuresFromStep1)):
            for Step1 in range(len(featuresFromStep1[dataFromStepNMinus1])):
                writer.writerow(featuresFromStep1[dataFromStepNMinus1][Step1])
        for dataFromStepNMinus2 in range(len(featuresFromStep2)):
            for Step2 in range(len(featuresFromStep2[dataFromStepNMinus2])):
                writer.writerow(featuresFromStep2[dataFromStepNMinus2][Step2])
        for dataFromStepNMinus3 in range(len(featuresFromStep3)):
            for Step3 in range(len(featuresFromStep3[dataFromStepNMinus3])):
                writer.writerow(featuresFromStep3[dataFromStepNMinus3][Step3])
        for dataFromStepNMinus4 in range(len(featuresFromStep4)):
            for Step4 in range(len(featuresFromStep4[dataFromStepNMinus4])):
                writer.writerow(featuresFromStep4[dataFromStepNMinus4][Step4])


if __name__ == '__main__':
    # INTRO: The key with data collection is to use a wide-range of songs with a wide range of sentiment. First we
    # add the simple cases of positive and negative songs, then finally add songs of positive and negative sentiment
    # for every genre. This will give a dataset that is mixed with every type of genre and every level of sentiment.

    # Features look like: [X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, y] = [Danceability, energy, key, loudness,
    # mode, speechiness, acousticness, instrumentalness, liveness, tempo, valence, sentiment]
    load_dotenv()
    featuresFromStep1 = []
    featuresFromStep2 = []
    featuresFromStep3 = []
    featuresFromStep4 = []

    # STEP 1. Initial positive sentiment songs:

    #  Songs from a "happy" genre (designed by Spotify & ranges from slightly positive to extremely positive sentiment).
    listOfSongID = getListOfSongIDFromGenre(50, "happy")
    # Get song's audio features and pass in expected sentiment.
    features = getListOfAudioFeatures(listOfSongID, 1)
    if features != -1:
        featuresFromStep1.append(features)
        print("Step 1 completed - 50 songs with positive sentiment added to data set. (1/5)")

    # STEP 2. Initial negative sentiment songs:

    # Songs from a "sad" genre (designed by Spotify & ranges from slightly negative to extremely negative sentiment).
    listOfSongID = getListOfSongIDFromGenre(50, "sad")
    # Get song's audio features and pass in expected sentiment.
    features = getListOfAudioFeatures(listOfSongID, 0)
    if features != -1:
        featuresFromStep2.append(features)
        print("Step 2 completed - 50 songs with negative sentiment added to data set. (2/5)")

    # Every genre Spotify uses (from https://developer.spotify.com/console/get-available-genre-seeds/)
    genres = ["acoustic", "afrobeat", "alt-rock", "alternative", "ambient", "anime", "black-metal", "bluegrass",
              "blues", "bossanova", "brazil", "breakbeat", "british", "cantopop", "chicago-house", "children", "chill",
              "classical", "club", "comedy", "country", "dance", "dancehall", "death-metal", "deep-house",
              "detroit-techno", "disco", "disney", "drum-and-bass", "dub", "dubstep", "edm", "electro", "electronic",
              "emo", "folk", "forro", "french", "funk", "garage", "german", "gospel", "goth", "grindcore", "groove",
              "grunge", "guitar", "happy", "hard-rock", "hardcore", "hardstyle", "heavy-metal", "hip-hop",
              "holidays", "honky-tonk", "house", "idm", "indian", "indie", "indie-pop", "industrial", "iranian",
              "j-dance", "j-idol", "j-pop", "j-rock", "jazz", "k-pop", "kids", "latin", "latino", "malay", "mandopop",
              "metal", "metal-misc", "metalcore", "minimal-techno", "movies", "mpb", "new-age", "new-release",
              "opera", "pagode", "party", "philippines-opm", "piano", "pop", "pop-film", "post-dubstep", "power-pop",
              "progressive-house", "psych-rock", "punk", "punk-rock", "r-n-b", "rainy-day", "reggae", "reggaeton",
              "road-trip", "rock", "rock-n-roll", "rockabilly", "romance", "sad", "salsa", "samba", "sertanejo",
              "show-tunes", "singer-songwriter", "ska", "sleep", "songwriter", "soul", "soundtracks", "spanish",
              "study", "summer", "swedish", "synth-pop", "tango", "techno", "trance", "trip-hop", "turkish",
              "work-out", "world-music"]

    # STEP 3. Wide range of positive sentiment songs for all types of genres:

    # A mix of words with slightly positive, somewhat positive, and extremely positive sentiment.
    positiveSentiments = ["nice", "go", "delight", "joy", "smile", "happy", "grin", "plus", "energy", "up", "best"]
    # Going through the nested loops below generates around 700 songs with positive sentiment.
    AllGenreWithPositiveSentimentSongID = []
    for genre in genres:
        for positiveSentiment in positiveSentiments:
            # Get song's audio features and pass in expected sentiment.
            features = getListOfAudioFeatures(getListOfSongIDFromGenreAndMood(1, genre, positiveSentiment), 1)
            if features != -1:
                print("Added song features of a song with a genre of " + genre + " and sentiment of " +
                      positiveSentiment + ".")
                AllGenreWithPositiveSentimentSongID.append(features[0])
    featuresFromStep3.append(AllGenreWithPositiveSentimentSongID)
    print("Step 3 completed - 700+ songs with positive sentiment added to data set. (3/5)")

    # STEP 4. Wide range of negative sentiment songs for all types of genres:

    # A mix of words with slightly negative, somewhat negative, and extremely negative sentiment
    negativeSentiments = ["sorrow", "regret", "down", "pain", "sad", "negative", "broken", "miserable", "numb",
                          "mourn", "broken"]
    # Going through the nested loops below generates around 700 songs with negative sentiment.
    AllGenreWithNegativeSentimentSongID = []
    for genre in genres:
        for negativeSentiment in negativeSentiments:
            # Get song's audio features and pass in expected sentiment.
            features = getListOfAudioFeatures(getListOfSongIDFromGenreAndMood(1, genre, negativeSentiment), 0)
            if features != -1:
                print("Added song features of a song with a genre of " + genre + " and sentiment of " +
                      negativeSentiment + ".")
                AllGenreWithNegativeSentimentSongID.append(features[0])
    featuresFromStep4.append(AllGenreWithNegativeSentimentSongID)
    print("Step 4 completed - 700+ songs with negative sentiment added to data set. (4/5)")

    # STEP 5. Write the collected data from web scraping into the dataset.csv file:

    writeFeaturesToFile(featuresFromStep1, featuresFromStep2, featuresFromStep3, featuresFromStep4, "dataset.csv")
    print("Step 5 completed - Wrote data from web scraping into dataset.csv. (5/5)")