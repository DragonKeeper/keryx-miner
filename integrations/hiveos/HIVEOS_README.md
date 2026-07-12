# HiveOS Packaging Notes

h-stop.sh and h-run.sh contain code to protect the miner's `models` directory when the custom miner install URL changes in the HiveOS config screen.

## What the stop script does

When HiveOS stops the custom miner during an upgrade or reinstall, `h-stop.sh` checks for the miner's local `models` directory and moves it out of the install path before HiveOS removes the old miner folder.

This prevents model downloads from being deleted every time the install URL changes.

## What the run script does

Before the miner starts, `h-run.sh` checks for a previously preserved shared `models` directory and moves it back into the current miner directory if needed.

This restores the model files after HiveOS has unpacked the new miner package.

## Tarball naming

Naming it like this with only 2 hyphens keeps the folder name that hive creates, short.  It creates a single folde each time named keryx-miner.  Every space after the second
hyphen should be an underscore.
The HiveOS release tarball should be named in this format:

```text
keryx-miner-v<version>_OPoI_hiveos.tar.gz
```

Use that naming pattern so HiveOS creates only one miner folder instead of creating a new folder for every version change. That keeps the install layout stable across updates and makes the model directory move logic work as intended.

## Summary

- `h-stop.sh` preserves `models` before HiveOS deletes the old install directory.
- `h-run.sh` restores `models` into the new install directory before startup.
- The tarball name should remain stable in structure across releases so HiveOS does not create multiple version folders.
