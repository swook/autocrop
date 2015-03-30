#!/bin/bash

BIN=../src/retarget

if [ ! -f "$BIN" ]; then
	echo "Error: Binary $BIN does not exist."
	exit
fi

if [ ! "$1" ]; then
	echo "$0 <folder-path>"
	exit
fi

DIR="$1/"

if [ ! -d "$1" ]; then
	echo "Error: directory $1 does not exist."
	exit
fi

for f in $(ls $DIR); do
	IN="$f"
	OUT="${f%%.*}_saliency.jpg"
	$BIN --headless "$DIR$IN" -o "$DIR$OUT"
done

