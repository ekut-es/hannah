echo -n "Are you sure that you have 125GB free space on Disk (y/n)? "
read answer

if [ "$answer" != "${answer#[Yy]}" ] ;then

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

    mkdir -p FSDKaggle
    pushd FSDKaggle

    wget https://zenodo.org/record/2552860/files/FSDKaggle2018.audio_test.zip
    unzip FSDKaggle2018.audio_test.zip
    rm FSDKaggle2018.audio_test.zip

    wget https://zenodo.org/record/2552860/files/FSDKaggle2018.audio_train.zip
    unzip FSDKaggle2018.audio_train.zip
    rm FSDKaggle2018.audio_train.zip

    wget https://zenodo.org/record/2552860/files/FSDKaggle2018.meta.zip
    unzip FSDKaggle2018.meta.zip
    rm FSDKaggle2018.meta.zip
    popd

    mkdir -p FSDnoisy
    pushd FSDnoisy
    wget https://zenodo.org/record/2529934/files/FSDnoisy18k.audio_test.zip
    unzip FSDnoisy18k.audio_test.zip
    rm FSDnoisy18k.audio_test.zip
    wget https://zenodo.org/record/2529934/files/FSDnoisy18k.audio_train.zip
    unzip FSDnoisy18k.audio_train.zip
    rm FSDnoisy18k.audio_train.zip
    wget https://zenodo.org/record/2529934/files/FSDnoisy18k.meta.zip
    unzip FSDnoisy18k.meta.zip
    rm FSDnoisy18k.meta.zip
    popd

    popd

    mkdir -p speech_files

    pushd speech_files
    wget https://zeos.ling.washington.edu/corpora/UWNU/uwnu-v2.tar.gz
    tar xvzf  uwnu-v2.tar.gz
    rm uwnu-v2.tar.gz

    mkdir -p mozilla
    pushd mozilla
    #dataset version 2020-06-22
    mkdir -p en
    pushd en
    wget https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5.1-2020-06-22/en.tar.gz
    tar xvzf en.tar.gz
    rm en.tar.gz
    popd

    mkdir de
    pushd de
    wget https://cdn.commonvoice.mozilla.org/cv-corpus-5.1-2020-06-22/de.tar.gz
    tar xvzf de.tar.gz
    rm de.tar.gz
    popd

    mkdir -p fr
    pushd fr
    wget https://cdn.commonvoice.mozilla.org/cv-corpus-5.1-2020-06-22/fr.tar.gz
    tar xvzf fr.tar.gz
    rm fr.tar.gz
    popd

    mkdir -p es
    pushd es
    wget https://cdn.commonvoice.mozilla.org/cv-corpus-5.1-2020-06-22/es.tar.gz
    tar xvzf es.tar.gz
    rm es.tar.gz
    popd

    mkdir -p it
    pushd it
    wget https://cdn.commonvoice.mozilla.org/cv-corpus-5.1-2020-06-22/it.tar.gz
    tar xvzf it.tar.gz
    rm it.tar.gz
    popd

    popd
    popd

    #python3 split_vad_data_balanced_extended.py
    #python3 downsample.py

else
    echo "then please choose the right place for the project!"
fi

