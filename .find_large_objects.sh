#!/bin/bash

#This scripts finds large objects that might accidentally be added to the repository and stores them in largeobjects.txt

git rev-list --objects --all | sort -k 2 > allfileshas.txt
git gc && git verify-pack -v .git/objects/pack/pack-*.idx | egrep "^\w+ blob\W+[0-9]+ [0-9]+ [0-9]+$" | sort -k 3 -n -r > bigobjects.txt
for SHA in `cut -f 1 -d\  < bigobjects.txt`; do
    echo $(grep $SHA bigobjects.txt) $(grep $SHA allfileshas.txt) | awk '{print $1,$3,$7}' >> bigtosmall.txt
    done;
mv bigtosmall.txt largeobjects.txt
rm allfileshas.txt bigobjects.txt 

