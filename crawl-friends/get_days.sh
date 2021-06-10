#!/bin/bash

files=$(ls | egrep '(real|bot).csv')
min_day=$((24 * 60))

for file in $files
do
	no_friends=$(cat $file | wc -l)
	echo $no_friends '/' $min_day $((no_friends / min_day))
done

echo $files
