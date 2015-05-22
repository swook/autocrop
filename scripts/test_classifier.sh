#!/bin/bash

# Go to where the script is
cd $(dirname "${BASH_SOURCE[0]}")

BUILD="../build"
BIN="$BUILD/classifier"
DATA="../datasets"

CHEN="$DATA/Chen"
CHEN_BAD="$CHEN/image"
echo; echo "Expect to be bad"; $BIN --headless "$CHEN_BAD/1702222221_Large.jpg"
echo; echo "Expect to be bad"; $BIN --headless "$CHEN_BAD/563030678_Large.jpg"
echo; echo "Expect to be bad"; $BIN --headless "$CHEN_BAD/917643980_Large.jpg"
echo; echo "Expect to be bad"; $BIN --headless "$CHEN_BAD/IMG_0350.jpg"
echo; echo "Expect to be bad"; $BIN --headless "$CHEN_BAD/206067269_Large.jpg"
echo; echo "Expect to be bad"; $BIN --headless "$CHEN_BAD/428858104_Large.jpg"
echo; echo "Expect to be bad"; $BIN --headless "$CHEN_BAD/990740513_Large.jpg"
echo; echo "Expect to be bad"; $BIN --headless "$CHEN_BAD/41224665_Large.jpg"
echo; echo "Expect to be bad"; $BIN --headless "$CHEN_BAD/944545105_Large.jpg"

CHEN_GOOD="$CHEN/analysis/good_crops"
echo; echo "Expect to be good"; $BIN --headless "$CHEN_GOOD/428858104_Large_10.jpg"
echo; echo "Expect to be good"; $BIN --headless "$CHEN_GOOD/990740513_Large_1.jpg"
echo; echo "Expect to be good"; $BIN --headless "$CHEN_GOOD/41224665_Large_9.jpg"
echo; echo "Expect to be good"; $BIN --headless "$CHEN_GOOD/944545105_Large_1.jpg"
echo; echo "Expect to be good"; $BIN --headless "$CHEN_GOOD/395373405_Large_5.jpg"

#EARTHPORN="$DATA/EarthPorn"
#EARTHPORN_GOOD="$EARTHPORN"
#echo; echo "Expect to be good"; $BIN --headless "$EARTHPORN_GOOD/0001.jpg"
#echo; echo "Expect to be good"; $BIN --headless "$EARTHPORN_GOOD/0011.jpg"
#echo; echo "Expect to be good"; $BIN --headless "$EARTHPORN_GOOD/0111.jpg"
#echo; echo "Expect to be good"; $BIN --headless "$EARTHPORN_GOOD/0311.jpg"

