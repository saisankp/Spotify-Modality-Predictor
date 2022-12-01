#!/bin/bash

cd src

touch .env

read -p "What is your Spotify API Key?: "

echo "SPOTIFY_API_TOKEN='$REPLY'" > .env