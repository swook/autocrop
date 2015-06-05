#!/bin/bash

wget -Nnv http://files.swook.net/autocrop/datasets/Michael.zip

unzip -oqq Michael.zip -d tmp
mv tmp/Michael/* .

rm -rf tmp

