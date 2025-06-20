ifndef VERSION_MK #include guard
# copied and slightly adapted from:
# https://stackoverflow.com/questions/44038428/include-git-commit-hash-and-or-branch-name-in-c-c-source/44038455#44038455
VERSION_MK=1
GIT_HASH=$$(git rev-parse HEAD)
COMPILE_TIME=$$(date -u +'%Y-%m-%dT%H:%M:%SZ')
GIT_BRANCH=$$(git branch --show-current)
export VERSION_FLAGS=-DGIT_HASH="\"$(GIT_HASH)\"" -DCOMPILE_TIME="\"$(COMPILE_TIME)\"" -DGIT_BRANCH="\"$(GIT_BRANCH)\""
endif # VERSION_MK
