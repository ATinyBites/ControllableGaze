# export CUDA_VISIBLE_DEVICES=3
# noglass=(2 3 4 5 6 7 9 13 14)
# noglass=(15 18 19 20 21 24 27 29 30)
# noglass=(31 33 35 38 40 42 44 45)
# noglass=(47 48 49 51 52 53 54 56)
# noglass=(2 3 4 5 6 7 9 13 14 15 18 19 20 21 24 27 29)
# noglass=(30 31 33 35 38 40 42 44 45 47 48 49 51 52 53 54 56)
# noglass=(49)
# noglass=(4 7 18 27 33)
noglass=(2 3 4 5 6 7 9 13 14 15 18 19 20 21 24 27 29 30 31 33 35 38 40 42 44 45 47 48 49 51 52 53 54 56)

for i in "${noglass[@]}"; do
    echo $i
    if ((i<10)); then
        python simulate.py --model_path output/columbia_0${i} --dataset_type columbia --data_size 1500 --output_h5 --output_h5_path synthetic_dataset/columbia
    else
        python simulate.py --model_path output/columbia_${i} --dataset_type columbia --data_size 1500  --output_h5 --output_h5_path synthetic_dataset/columbia
    fi
done 