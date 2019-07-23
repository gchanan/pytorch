#GIT_CHECKOUT="/home/gchanan/_pytorch"
BASE_VERSION="origin/v1.1.0"
NEW_VERSION=d8220b059901450bb138fbf27dadda8ff6965819

#GIT_WORK_TREE=$GIT_CHECKOUT

MERGE_BASE=`git merge-base $BASE_VERSION $NEW_VERSION`
echo $MERGE_BASE

git log --reverse --oneline ${MERGE_BASE}..${NEW_VERSION} | cut -d " " -f 1 > commit_list.txt
