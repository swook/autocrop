#!/bin/bash

for dir in $(find . -maxdepth 1 -mindepth 1 -type d); do
	cd "$dir"

	if [ -x "get_dataset.sh" -o -x "get_dataset.py" ]; then
		echo; echo "> Getting dataset for path \"$dir\""
		./get_dataset.*
	fi

	cd ..
done
