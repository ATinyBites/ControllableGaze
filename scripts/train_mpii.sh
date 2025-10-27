# export CUDA_VISIBLE_DEVICES=0
for i in {0..14}; do
        if ((i<10)); then
            python train_avatar.py -s dataset/mpii/mpii_0${i} -m output/mpii_0${i}  --bind_to_mesh --sh_degree=0 --load_light 
        else
            python train_avatar.py -s dataset/mpii/mpii_${i} -m output/mpii_${i} --bind_to_mesh --sh_degree=0 --load_light
        fi
done 
