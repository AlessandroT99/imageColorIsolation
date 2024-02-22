#!/bin/bash

#nTest=ls | wc -l
#((nTest=nTest-2))
# Counting number of directories
shopt -s nullglob
numdirs=(*/)
numdirs=${#numdirs[@]}
((numdirs -= 1))

echo "Video creation for all the $numdirs tests"

for cnt in {1..9}
do
	cd "./P_0000"$cnt
	printf "\nConverting files for test number $cnt... "
	if ! [ -f *.mp4 ]; then
		ffmpeg -framerate 30 -i "%08d.ppm" -vcodec libx264 -crf 25 -pix_fmt yuv420p "P$cnt.mp4"
		echo "Conversion for test number $cnt completed."
	else
		echo "File for test number $cnt already created."
	fi
	cd ..
done

if [$numdirs > 9]; then 
	for cnt in {10..23}
	do
		cd "./P_000"$cnt
		printf "\nConverting files for test number $cnt... "
		if ! [ -f *.mp4 ]; then
			ffmpeg -framerate 30 -i "%08d.ppm" -vcodec libx264 -crf 25 -pix_fmt yuv420p "P$cnt.mp4"
			echo "Conversion for test number $cnt completed."
		else
			echo "File for test number $cnt already created."
		fi
		cd ..
	done
fi

echo "Video creation complete."

