#!/bin/bash

py=~/miniconda3/bin/python

for threshold in 5 10 20; do
	mkdir -p sheets_${threshold}
	$py read_result.py -i ./RGB --threshold ${threshold} -o sheets_${threshold} -c rgb --taxon specific_epithet
done
