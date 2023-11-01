cd ../

# best for bal

# 33 0.732125 0.732125 0.6780619268754975 0.41880919110272347 ../fairlib_uncertainty/results/dev/Moji/moji_mlp_hp_search/moji_bal_fairlib_mlp_comp_mdto_hypopt_deb_vanilla_128_0.003_0.1_5304/models/BEST_checkpoint.pth.tar





# BTEO model
for bs in 128
do
    for lr in 0.003
    do
        for drop in 0.1
        do
            python fairlib --dataset Moji --emb_size 2304 --num_classes 2 --num_groups 2 --batch_size $bs --lr $lr --dropout $drop --early_stopping_criterion max_balanced_dto --BT Reweighting --BTObj EO --data_dir ../fl_scripts/data/deepmoji --base_seed 5304 --exp_id "moji_mlp_hp_search/moji_bal_fairlib_mlp_comp_mdto_hypopt_deb_bteo_${bs}_${lr}_${drop}_5304"
        done
    done
done
# Now we run all methods only for the best params from the standard model
# adv model
for bs in 128
do
    for lr in 0.003
    do
        for drop in 0.1
        do
            for adv in 0.0001 0.001 0.01 0.1 1 10 100
            do
                python fairlib --dataset Moji --emb_size 2304 --num_classes 2 --num_groups 2 --batch_size $bs --lr $lr --dropout $drop --early_stopping_criterion max_balanced_dto --adv_debiasing --adv_lambda $adv --data_dir ../fl_scripts/data/deepmoji --base_seed 5304 --exp_id "moji_mlp_hp_search/moji_bal_fairlib_mlp_comp_mdto_hypopt_deb_adv_${bs}_${lr}_${drop}_${adv}_5304"
            done
        done
    done
done
# dadv model
for bs in 128
do
    for lr in 0.003
    do
        for drop in 0.1
        do
            for adv in 0.0001 0.001 0.01 0.1 1 10 100
            do
                python fairlib --dataset Moji --emb_size 2304 --num_classes 2 --num_groups 2 --batch_size $bs --lr $lr --dropout $drop --early_stopping_criterion max_balanced_dto --adv_debiasing --adv_lambda $adv --adv_num_subDiscriminator 3 --adv_diverse_lambda $adv --data_dir ../fl_scripts/data/deepmoji --base_seed 5304 --exp_id "moji_mlp_hp_search/moji_bal_fairlib_mlp_comp_mdto_hypopt_deb_dadv_${bs}_${lr}_${drop}_${adv}_${adv}_5304"
            done
        done
    done
done
# fairbatch model
for bs in 128
do
    for lr in 0.003
    do
        for drop in 0.1
        do
            for param in 0.0001 0.001 0.01 0.05 0.1 0.5 1
            do
                python fairlib --dataset Moji --emb_size 2304 --num_classes 2 --num_groups 2 --batch_size $bs --lr $lr --dropout $drop --early_stopping_criterion max_balanced_dto --DyBT FairBatch --DyBTObj stratified_y --DyBTalpha $param --data_dir ../fl_scripts/data/deepmoji --base_seed 5304 --exp_id "moji_mlp_hp_search/moji_bal_fairlib_mlp_comp_mdto_hypopt_deb_fairbatch_${bs}_${lr}_${drop}_${param}_5304"
            done
        done
    done
done

# GDdiff model
for bs in 128
do
    for lr in 0.003
    do
        for drop in 0.1
        do
            for param in 0.0001 0.5 0.001 0.01 0.1 0.0 1.0
            do
                python fairlib --dataset Moji --emb_size 2304 --num_classes 2 --num_groups 2 --batch_size $bs --lr $lr --dropout $drop --early_stopping_criterion max_balanced_dto --DyBT GroupDifference --DyBTObj EO --DyBTalpha $param --data_dir ../fl_scripts/data/deepmoji --base_seed 5304 --exp_id "moji_mlp_hp_search/moji_bal_fairlib_mlp_comp_mdto_hypopt_deb_gddiff_${bs}_${lr}_${drop}_${param}_5304"
            done
        done
    done
done
# BTJ model
for bs in 128
do
    for lr in 0.003
    do
        for drop in 0.1
        do
            python fairlib --dataset Moji --emb_size 2304 --num_classes 2 --num_groups 2 --batch_size $bs --lr $lr --dropout $drop --early_stopping_criterion max_balanced_dto --BT Reweighting --BTObj joint --data_dir ../fl_scripts/data/deepmoji --base_seed 5304 --exp_id "moji_mlp_hp_search/moji_bal_fairlib_mlp_comp_mdto_hypopt_deb_btj_${bs}_${lr}_${drop}_5304"
        done
    done
done
# INLP model
# as INLP is postprocessing method, we will only make 4 runs for the best vanilla model
# here we tune bool params, so params are tuned inside one loop
for bs in 128
do
    for lr in 0.003
    do
        for drop in 0.1
        do
            # common
            python fairlib --dataset Moji --emb_size 2304 --num_classes 2 --num_groups 2 --batch_size $bs --lr $lr --dropout $drop --early_stopping_criterion max_balanced_dto --INLP --INLP_min_acc 0.5 --data_dir ../fl_scripts/data/deepmoji --base_seed 5304 --exp_id "moji_mlp_hp_search/moji_bal_fairlib_mlp_comp_mdto_hypopt_deb_inlp_${bs}_${lr}_${drop}_0_0_5304"
            # by_class
            python fairlib --dataset Moji --emb_size 2304 --num_classes 2 --num_groups 2 --batch_size $bs --lr $lr --dropout $drop --early_stopping_criterion max_balanced_dto --INLP --INLP_min_acc 0.5 --INLP_by_class --data_dir ../fl_scripts/data/deepmoji --base_seed 5304 --exp_id "moji_mlp_hp_search/moji_bal_fairlib_mlp_comp_mdto_hypopt_deb_inlp_${bs}_${lr}_${drop}_1_0_5304"
            # reweighting
            python fairlib --dataset Moji --emb_size 2304 --num_classes 2 --num_groups 2 --batch_size $bs --lr $lr --dropout $drop --early_stopping_criterion max_balanced_dto --INLP --INLP_min_acc 0.5 --INLP_discriminator_reweighting True --data_dir ../fl_scripts/data/deepmoji --base_seed 5304 --exp_id "moji_mlp_hp_search/moji_bal_fairlib_mlp_comp_mdto_hypopt_deb_inlp_${bs}_${lr}_${drop}_0_1_5304"
            # reweighting and by_class
            python fairlib --dataset Moji --emb_size 2304 --num_classes 2 --num_groups 2 --batch_size $bs --lr $lr --dropout $drop --early_stopping_criterion max_balanced_dto --INLP --INLP_min_acc 0.5 --INLP_by_class --INLP_discriminator_reweighting True --data_dir ../fl_scripts/data/deepmoji --base_seed 5304 --exp_id "moji_mlp_hp_search/moji_bal_fairlib_mlp_comp_mdto_hypopt_deb_inlp_${bs}_${lr}_${drop}_1_1_5304"
        done
    done
done
