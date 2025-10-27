
# camtype=("webcam_c")
camtype=("basler")
for cam in "${camtype[@]}"; do
    for i in {1..1}; do
            python simulate.py --model_path output/EVE_val0${i}_${cam}_full_v2 --white_background --dataset_type eve --data_size 100 
    done
done