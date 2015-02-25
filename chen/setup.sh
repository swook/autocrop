#!/bin/bash

wget -Nnv http://www.cs.dartmouth.edu/%7Echenfang/proj_page/FLMS_mm14/data/radomir500_gt/release_data.tar
wget -Nnv http://www.cs.dartmouth.edu/%7Echenfang/proj_page/FLMS_mm14/data/radomir500_image/image.tar

tar xf release_data.tar
tar xf image.tar
