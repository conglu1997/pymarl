#!/bin/bash
rsync -chavzP --stats ${1}@${2}:~/pymarl/results ./
