#!/bin/bash

for dir in $(find . -maxdepth 1 -mindepth 1 -type d); do
	cd "$dir"
	if [ -x "get_dataset.sh" ]; then
		echo "Getting dataset for path \"$dir\""
		bash get_dataset.sh
	fi
	cd ..
done
