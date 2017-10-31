#!/bin/bash

branch_name="$(git symbolic-ref HEAD 2>/dev/null)" ||
branch_name="(unnamed branch)"     # detached HEAD
branch_name=${branch_name##refs/heads/}
echo "Current branch", $branch_name

mkdir -p ../comparisonDir/$branch_name && cp -r * ../comparisonDir/$branch_name

branches=()
eval "$(git for-each-ref --shell --format='branches+=(%(refname))' refs/heads/)"

for branch in "${branches[@]}"; do
    b=${branch##refs/heads/}
    if [ $b != $branch_name ]
    then
    	git checkout $b 
    	meld . ../comparisonDir/$branch_name
    	git add --all
    	git commit -m
    fi
done
git checkout $branch_name