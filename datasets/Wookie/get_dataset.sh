#!/bin/bash

wget -Nnv http://files.swook.net/autocrop/datasets/Wookie.zip

unzip -oqq Wookie.zip -d tmp
mv tmp/Wookie/* .

rm -rf tmp

