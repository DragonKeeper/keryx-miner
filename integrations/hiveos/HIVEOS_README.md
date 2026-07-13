
## What the run script now does

Before the miner starts, `h-run.sh` checks for a previously preserved shared `models` directory and moves it back into the current miner directory if needed.

This restores the model files after HiveOS has unpacked the new miner package.

## Tarball naming

Naming it like this with only 2 hyphens keeps the folder name that hive creates, short.  It creates a single folder each time named keryx-miner.  Every space after the second
hyphen should be an underscore.
The HiveOS release tarball should be named in this format:

```text
keryx-miner-v<version>_OPoI_hiveos.tar.gz
```

Use that naming pattern so HiveOS creates only one miner folder instead of creating a new folder for every version change. That keeps the install layout stable across updates and makes the model directory move logic work as intended.

## Summary

- The tarball name should remain stable in structure across releases so HiveOS does not create multiple version folders.
- The first time going from current naming format to the new format, a user( from a hiveos shell ) will have to run the following command to get the shell script that 
    will run `h-stop.sh` and move the models folder up one directory level before changing the Install URL to the new version in their custom config screen so that it will preserve their models.  Commands are as follows:
    ```bash 
    cd /hive/miners/custom/keryx-miner-v0.3.6-OPoI/ \
    wget -O - https://keryx-labs.com/pre-hive-upgrade.sh | bash
    ```
