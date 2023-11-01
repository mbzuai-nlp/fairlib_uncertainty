cd ../

# vanilla model
for bs in 64 128 256 512 1024
do
    for lr in 0.01 0.005 0.003 0.001 0.0005 0.0001
    do
        for drop in 0.0 0.1 0.2 0.3
        do
            python fairlib --dataset Moji --emb_size 2304 --num_classes 2 --num_groups 2 --batch_size $bs --lr $lr --dropout $drop --data_dir ../fl_scripts/data/deepmoji --unbalance_test --base_seed 5304 --exp_id "moji_mlp_hp_search/moji_unbal_fairlib_mlp_comp_mdto_hypopt_deb_vanilla_${bs}_${lr}_${drop}_5304" --device_id 1
        done
    done
done

# best for unbal
# 92 0.8379351740696278 0.8379351740696278 0.6285978335216379 0.4052216394378055 ../fairlib_uncertainty/results/dev/Moji/moji_mlp_hp_search/moji_unbal_fairlib_mlp_comp_mdto_hypopt_deb_vanilla_512_0.0001_0.0_5304/models/BEST_checkpoint.pth.tar

