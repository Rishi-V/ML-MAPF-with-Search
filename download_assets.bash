if [ ! -d "data" ]; then
    mkdir data
fi
cd data

# Download maps
if [ ! -d "mapf-map" ]; then
    wget https://movingai.com/benchmarks/mapf/mapf-map.zip
    unzip mapf-map.zip -d mapf-map
fi

# Download random scens; note other scens are available to try too
if [ ! -d "mapf-scen-random" ]; then
    wget https://movingai.com/benchmarks/mapf/mapf-scen-random.zip
    unzip mapf-scen-random.zip
    mv scen-random mapf-scen-random # rename for consistency
fi

# Download backward dijkstras npzs
gdown https://drive.google.com/drive/folders/1S3md2fHR2cahc_yoeJxNeKl_gti-JU0h?usp=drive_link -O constant_npzs/ --folder --continue

# Download all_maps npzs
gdown https://drive.google.com/uc?id=1vky4vcVkXvLbyKAM7WGJ96U698OvywwM -O constant_npzs/ --continue

# Download model
if [ ! -d "model" ]; then
    mkdir model
fi
gdown https://drive.google.com/uc?id=1Xei6-s92bsLs44ihkriZ2nOi8z1r7RQV -O model/ --continue

# Create logs folder
cd ..
if [ ! -d "logs" ]; then
    mkdir logs  # Optional, recommended for consistency with batch_runner.py
fi