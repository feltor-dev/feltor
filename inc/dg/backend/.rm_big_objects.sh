#!/bin/bash

#this script is an EMERGENCY script and permanently removes objects from the git repository including the history
# its intention is to reduce the repo size after a large file (binary etc.) has been accidentally added and pushed
# BE VERY CAREFUL WITH THIS SINCE FILES WILL BE PERMANENTLY REMOVED
# ALSO, THE HISTORY WILL BE CHANGED WHICH WILL TOTALLY UPSET EVERYONE WHO EVER CLONED THE REPOSITORY
# after execution it's as if these objects never existed.
BIG_OBJECTS="src/feltorSesol/core src/feltorShw/core src/feltorShw/feltor src/feltorSesol/feltor"

echo 'Removing '$BIG_OBJECTS''

git filter-branch --prune-empty --index-filter "git rm -rf --cached --ignore-unmatch $BIG_OBJECTS" --tag-name-filter cat -- --all
#up to now the files can still be recovered


rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now
git gc --aggressive --prune=now

# use git push origin --force --all to tell github that these objects vanished
# All other teammates should then freshly clone the repository from github or run this script since any push will readd the objects again

