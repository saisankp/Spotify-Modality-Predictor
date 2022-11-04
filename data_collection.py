import os
import requests
from dotenv import load_dotenv
import csv


# You can use the ID retrieved at https://open.spotify.com/track/{Enter-ID-Here-Without-Brackets} to listen to the song.
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
    for track in range(numberOfTracks):
        listOfSongID.append(songs["tracks"]["items"][track]["id"])
    return listOfSongID


def getListOfAudioFeatures(listOfSongID, sentiment):
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
    load_dotenv()
    # [X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, y] = [Danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, tempo, sentiment]
    featuresFromStep1 = []
    featuresFromStep2 = []
    featuresFromStep3 = []
    featuresFromStep4 = []
    # The key with data collection is to use a wide-range of songs with a wide range of sentiment. First we
    # add the simple cases of positive and negative songs, then finally add songs of positive and negative sentiment
    # for every genre. This will give a dataset that is mixed with every type of genre and every level of sentiment.

    # 1. Initial positive sentiment songs:

    #  Songs from a "happy" genre (designed by Spotify & ranges from slightly positive to extremely positive sentiment)
    listOfSongID = getListOfSongIDFromGenre(50, "happy")
    # Get song's audio features and pass in expected sentiment
    featuresFromStep1.append(getListOfAudioFeatures(listOfSongID, 1))

    print("Step 1 completed - 50 songs with positive sentiment added to data set")

    # 2. Initial negative sentiment songs

    # Get songs from a "sad" genre (designed by Spotify & ranges from slightly negative to extremely negative sentiment)
    listOfSongID = getListOfSongIDFromGenre(50, "sad")
    # Get song's audio features and pass in expected sentiment
    featuresFromStep2.append(getListOfAudioFeatures(listOfSongID, 0))

    print("Step 2 completed - 50 songs with negative sentiment added to data set")

    # Genres from Spotify
    genres = ["acoustic", "alt-rock", "alternative", "ambient", "anime", "black-metal",
              "blues",
              "brazil", "british",  "chill", "classical",
              "club", "comedy", "country", "dance", "dancehall", "death-metal",
              "disney", "dub", "edm", "electro", "electronic", "emo", "folk",
              "french",
              "funk", "german", "gospel", "goth", "grindcore",  "grunge", "guitar", "hard-rock",
              "hardcore",
               "heavy-metal", "hip-hop", "house",  "indie",
              "indie-pop",
              "industrial", "j-pop", "j-rock", "jazz",  "latin",
              "malay", "metal", "metalcore", "new-age", "piano", "pop",
              "power-pop", "psych-rock", "punk", "punk-rock", "r-n-b", "reggae",
              "rock",
              "singer-songwriter", "ska", "sleep", "songwriter", "soul", "spanish",
              "swedish"]

    # 3. Wide range of positive sentiment songs for all types of genres

    # A mix of words with slightly positive, somewhat positive, and extremely positive sentiment
    positiveSentiments = ["nice", "go", "delight", "joy", "smile", "happy", "grin", "plus", "energy",
                          "up", "best"]
    # Going through the nested loops gives 1364 songs
    count = 0
    AllGenreWithPositiveSentimentSongID = []
    for genre in genres:
        for positiveSentiment in positiveSentiments:
            search = genre + " " + positiveSentiment
            # print(getListOfAudioFeatures(getListOfSongIDFromGenre(1, search), 1))
            count = count+1
            AllGenreWithPositiveSentimentSongID.append(
                getListOfAudioFeatures(getListOfSongIDFromGenre(1, search), 1)[0])
    # Get song's audio features and pass in expected sentiment
    featuresFromStep3.append(AllGenreWithPositiveSentimentSongID)
    print(count)

    print("Step 3 completed - X songs with positive sentiment added to data set")

    # 4. Wide range of negative sentiment songs for all types of genres

    # A mix of words with slightly negative, somewhat negative, and extremely negative sentiment
    negativeSentiments = ["sorrow", "regret", "down", "pain", "sad", "negative", "broken", "miserable", "numb",
                          "mourn", "broken"]
    # Going through the nested loops gives 1364 songs
    AllGenreWithNegativeSentimentSongID = []
    for genre in genres:
        for negativeSentiment in negativeSentiments:
            search = genre + " " + negativeSentiment
            AllGenreWithNegativeSentimentSongID.append(
                getListOfAudioFeatures(getListOfSongIDFromGenre(1, search), 0)[0])
    # Get song's audio features and pass in expected sentiment
    featuresFromStep4.append(AllGenreWithNegativeSentimentSongID)

    print("Step 4 completed - 50 songs with negative sentiment added to data set")

    writeFeaturesToFile(featuresFromStep1, featuresFromStep2, featuresFromStep3, featuresFromStep4, "dataset.csv")

    print("Step 5 completed - Wrote data from web scraping into CSV.")