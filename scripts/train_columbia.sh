# export CUDA_VISIBLE_DEVICES=0
# noglass=(2 3 4 5 6 7 9 13 14)
# noglass=(15 18 19 20 21 24 27 29 30)
# noglass=(31 33 35 38 40 42 44 45)
# noglass=(47 48 49 51 52 53 54 56)
noglass=(2 3 4 5 6 7 9 13 14 15 18 19 20 21 24 27 29 30 31 33 35 38 40 42 44 45 47 48 49 51 52 53 54 56)
for i in "${noglass[@]}"; do
    if ((i<10)); then
        python train_avatar.py -s dataset/columbia/columbia_0${i} -m output/columbia_0${i}   --bind_to_mesh 
     
    else
        python train_avatar.py -s dataset/columbia/columbia_${i} -m output/columbia_${i}   --bind_to_mesh
    fi
done