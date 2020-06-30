mkdir -p noise_files

pushd noise_files
wget https://zenodo.org/record/400515/files/TUT-acoustic-scenes-2017-development.audio.{1,2,3,4,5,6,7,8,9,10}.zip
unzip  -P pass '*.zip'

#rm  -f TUT-acoustic-scenes-2017-development.audio.1.zip
#rm  -f TUT-acoustic-scenes-2017-development.audio.2.zip
#rm  -f TUT-acoustic-scenes-2017-development.audio.3.zip
#rm  -f TUT-acoustic-scenes-2017-development.audio.4.zip
#rm  -f TUT-acoustic-scenes-2017-development.audio.5.zip
#rm  -f TUT-acoustic-scenes-2017-development.audio.6.zip
#rm  -f TUT-acoustic-scenes-2017-development.audio.7.zip
#rm  -f TUT-acoustic-scenes-2017-development.audio.8.zip
#rm  -f TUT-acoustic-scenes-2017-development.audio.9.zip
#rm  -f TUT-acoustic-scenes-2017-development.audio.10.zip

popd

mkdir -p speech_files

pushd speech_files

#currently not available
wget https://zeos.ling.washington.edu/corpora/UWNU/uwnu-v2.tar.gz --no-check-certificate
tar -xvzf  uwnu-v2.tar.gz
#rm -f uwnu-v2.tar.gz

popd 

python3 split_vad_data_balanced.py
python3 downsample.py
