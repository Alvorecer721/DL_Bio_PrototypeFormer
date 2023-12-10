#!/bin/sh

cd fewshotbench
python -m pip install gdown
python -m pip install -r requirements.txt

gdown --id 1a3IFmUMUXBH8trx_VWKZEGteRiotOkZS
unzip -q swissprot.zip
rm -rf swissprot.zip
mv data/swissprot/go-basic.obo ./