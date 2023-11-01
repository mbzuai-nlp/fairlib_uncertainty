cd ../

# vanilla model
for bs in 64 128 256 512 1024
do
    for lr in 0.01 0.005 0.003 0.001 0.0005 0.0001
    do
        for drop in 0.0 0.1 0.2 0.3
        do
            python fairlib --dataset Moji --emb_size 2304 --num_classes 2 --num_groups 2 --batch_size $bs --lr $lr --dropout $drop --data_dir ../fl_scripts/data/deepmoji --base_seed 5304 --exp_id "moji_mlp_hp_search/moji_bal_fairlib_mlp_comp_mdto_hypopt_deb_vanilla_${bs}_${lr}_${drop}_5304"
        done
    done
done

# best for bal

# 33 0.732125 0.732125 0.6780619268754975 0.41880919110272347 ../fairlib_uncertainty/results/dev/Moji/moji_mlp_hp_search/moji_bal_fairlib_mlp_comp_mdto_hypopt_deb_vanilla_128_0.003_0.1_5304/models/BEST_checkpoint.pth.tar

