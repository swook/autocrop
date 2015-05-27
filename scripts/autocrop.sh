#!/bin/bash

SCRIPTDIR="$(dirname "${BASH_SOURCE[0]}")"
BIN="$SCRIPTDIR/../build/autocrop"

# Make autocropped directory
OUTD="autocropped/"
if [ ! -d "$OUTD" ]; then
	mkdir "$OUTD"
fi

# If file
if [ -f "$1" ]; then
	$BIN "$1" -o "$OUTD/${1##*/}"

# If folder
elif [ -d "$1" ]; then
	for f in $1/*.{png,jpg}; do
		if [ ! -f "$f" ]; then
			continue
		fi

		$BIN "$f" -h -o "$OUTD/${f##*/}"
		echo "Autocropped $f"
	done
fi

