# Zprojection

This is thought of as a temporary repo.
So feel free to just commit whatever, without much concern for the repo's history.

## Goal

Investigate how useful it is to build a **Z-ranking**.
This is meant to be a monotonically increasing ranking of cuts.
For decreasing required efficiency, the cut extension that provides the best
purity is searched.

The Z refers to that fact that only remnants of the recoiling Z boson from a
Higgsstrahlung events are supposed to be used.

For more information, have a look at the notebooks.

## Setup

### Submodules

```sh
git clone --recurse-submodules -j8 git@github.com:LLR-ILD/Zprojection.git
# Or, if the repository was already cloned without the submodules:
git submodule update --init --recursive
```
