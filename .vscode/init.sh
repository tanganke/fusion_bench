#! /bin/bash
SCRIPT_DIR=$(cd $(dirname $0); pwd)

for file in launch.json settings.json
do
    if [ -f ${SCRIPT_DIR}/${file} ]; then
        echo "File ${file} already exists, skipping"
    else
        cp -v ${SCRIPT_DIR}/${file}.template ${SCRIPT_DIR}/${file}
    fi
done
