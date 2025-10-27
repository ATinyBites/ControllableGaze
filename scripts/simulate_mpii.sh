# export CUDA_VISIBLE_DEVICES=0
for i in {0..14}; do
    if ((i<10)); then
        python simulate.py --model_path output/mpii_0${i} --dataset_type mpii --data_size 3000   --load_light --anno --render_eye --output_h5 --output_h5_path synthetic_dataset/mpii
    else
        python simulate.py --model_path output/mpii_${i} --dataset_type mpii --data_size 3000   --load_light --anno --render_eye --output_h5 --output_h5_path synthetic_dataset/mpii
    fi
done