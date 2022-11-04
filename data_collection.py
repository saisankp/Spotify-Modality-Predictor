import os
import requests
from dotenv import load_dotenv


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
    if ("Invalid access token" in str(songs) or "The access token expired" in str(songs) or "Invalid limit" in str(songs)):
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
        songFeatures.append(sentiment)
        # songFeatures.append(((audioFeaturesForSong["audio_features"][0]["danceability"]) + (songFeatures.append(audioFeaturesForSong["audio_features"][0]["energy"]) + audioFeaturesForSong["audio_features"][0]["valence"]))/3)
        # songFeatures.append(audioFeaturesForSong["audio_features"][0]["valence"] + audioFeaturesForSong["audio_features"][0]["danceability"] + audioFeaturesForSong["audio_features"][0]["energy"])
        # songFeatures.append(audioFeaturesForSong["audio_features"][0]["valence"])
        # songFeatures.append(audioFeaturesForSong["audio_features"][0]["id"])
        listOfAudioFeatures.append(songFeatures)
    return listOfAudioFeatures


if __name__ == '__main__':
    load_dotenv()

    # Get songs from "happy" genre (designed by Spotify & ranges from slightly positive to extremely positive sentiment)
    listOfSongID = getListOfSongIDFromGenre(5, "happy")
    print(listOfSongID)
    # Get song's audio features and pass in expected sentiment
    print(getListOfAudioFeatures(listOfSongID, 1))

    # Get songs from "sad" genre (designed by Spotify & ranges from slightly negative to extremely negative sentiment)
    listOfSongID = getListOfSongIDFromGenre(5, "sad")
    print(listOfSongID)
    # Get song's audio features and pass in expected sentiment
    print(getListOfAudioFeatures(listOfSongID, 0))
