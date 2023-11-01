# How to run experiments

1. To reproduce paper results, one has to firstly run a hyperparameter search for a vanilla model with the following scripts: run_hp_search_bertweet_moji_unbal_vanilla.sh, run_hp_search_bertweet_moji_bal_vanilla.sh, run_hp_search_bert_bios_vanilla_cased_unbal_sub_prof_big.sh, run_hp_search_bert_bios_vanilla_cased_bal_sub_prof_big.sh, hp_search_mlp_moji_bal_vanilla.sh, hp_search_mlp_moji_unbal_vanilla.sh

2. After that one should run hyperparameter tuning for models with debiasing, using optimal training parameters from the first step of optimization. The scripts for tuning rest of the models: run_hp_search_bert_bios_debiasing_bal_sub_prof_big.sh, run_hp_search_bert_bios_debiasing_unbal_sub_prof_big.sh, run_hp_search_bertweet_moji_debiasing_unbal.sh, run_hp_search_bertweet_moji_debiasing_bal.sh, hp_search_mlp_moji_unbal.sh, hp_search_mlp_moji_bal.sh

3. To obtain results with UE, run following scripts: run_train_bertweet_opt_moji_no_sn_greedy_mdto_final_bal.sh, run_train_bertweet_opt_moji_no_sn_greedy_mdto_final_unbal.sh, run_train_bert_opt_bios_no_sn_greedy_mdto_final_bal_sub_prof_big.sh, run_train_bert_opt_bios_no_sn_greedy_mdto_final_unbal_sub_prof_big.sh, mlp_moji_unbal_mdto_opt.sh, mlp_moji_bal_mdto_opt.sh


4. To obtain results on OOD tasks simply run \*_ood.sh scripts.