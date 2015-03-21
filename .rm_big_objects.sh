#!/bin/bash

#this script permanently removes large objects from the git repository
# that have been accidentally included, 
# it also alters the history, it's as if the object never existed. 
# BE VERY CAREFUL WITH THIS SINCE FILES WILL BE REMOVED PERMANENTLY
BIG_OBJECTS="src/feltor2D/test.nc src/feltor2D/geometry.nc inc/dg/backend/dzs_t src/feltor2D/feltor inc/dg/backend/weights3d_t dg_lib/toefl_hpc src/heat/heat"

echo 'Removing '$BIG_OBJECTS''

git filter-branch --prune-empty --index-filter "git rm -rf --cached --ignore-unmatch $BIG_OBJECTS" --tag-name-filter cat -- --all
#up to now the files can still be recovered


rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now
git gc --aggressive --prune=now

# use git push origin --force --all to tell github that these objects vanished
# All other teammates should then freshly clone the repository from github since any push will readd the objects again

