import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import logging

try:
    from .src.base_options import BaseOptions
    from .src import networks
    from .src.utils import kfold_train_dev
except:
    from src.base_options import BaseOptions
    from src import networks
    from src.utils import kfold_train_dev

def main():
    options = BaseOptions()
    state = options.get_state()
    if state.cross_val:
        for i, (train_idx, dev_idx) in kfold_train_dev(state):
            args = {"cross_val_fold": i, 
                    "train_idx": train_idx,
                    "dev_idx": dev_idx}
            fold_options = BaseOptions()
            fold_state = fold_options.get_state(args=args, silence=True)
            
            # Init the model
            model = networks.get_main_model(fold_state)
            logging.info(f'Fold {i}: Model Initialized!')

            model.train_self()
            logging.info(f'Fold {i}: Model Trained!')

            if fold_state.INLP:
                logging.info(f'Fold {i}: Run INLP')
                from src.networks.INLP import get_INLP_trade_offs
                get_INLP_trade_offs(model, fold_state)

            logging.info(f'Fold {i}: Finished!')

    else:
        # Init the model
        model = networks.get_main_model(state)
        # state.opt.main_model = model
        logging.info('Model Initialized!')

        model.train_self()
        logging.info('Model Trained!')

        if state.INLP:
            logging.info('Run INLP')
            from src.networks.INLP import get_INLP_trade_offs
            get_INLP_trade_offs(model, state)

    logging.info('Finished!')

if __name__ == '__main__':
    main()
