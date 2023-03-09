"""Experiment with standard data sets, see experiments_ordinal_parallel7.py."""
import argparse, glob, os
import numpy as np
import pandas as pd
from datetime import datetime
from imblearn.metrics import geometric_mean_score
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from .. import factory

from ..utils import create_bags_with_multiple_prevalence
from ..metrics.multiclass import mean_absolute_error
from ..metrics.ordinal import emd, emd_distances, emd_score

BINS_GEN = 8
BINS_PDF_L2 = 4
BINS_PDF_EMD = 32
BINS_HDX = 8
COLUMNS = [
    'dataset',
    'method',
    'decomposer',
    'repxbag',
    'truth',
    'predictions',
    'mae',
    'mse',
    'emd',
    'emd_score'
]
DEFAULT_METHODS = [
    'CC',
    'AC_L2',
    'AC_Ord',
    'PCC',
    'PAC_L2',
    'EDX',
    'CvMy_Eu',
    'EDy_Eu',
    'EDy_EMD',
    'HDX',
    'HDy',
    'PDF_L2',
    'PDF_EMD',
]

def main(
        methods = DEFAULT_METHODS,
        seed = 2032,
        n_bags = 300,
        n_reps = 10,
        n_folds = 20,
        option = 'CV(DECOMP)',
        decomposer = 'Monotone', # or 'FHTree'
        n_jobs = -1,
        is_test_run = False,
        ):
    estimator = RandomForestClassifier(
        random_state = seed,
        class_weight = 'balanced'
    )
    estimator_grid = {
        'n_estimators': [100],
        'max_depth': [1, 5, 10, 15, 20, 25, 30],
        'min_samples_leaf': [1, 2, 5, 10, 20],
    }
    dataset_names = [
        'SWD',  #23.048
        'ESL',  # 5.395  -->5 clases: 3,4,5,6,7
        'LEV',  #11.023
        'cement_strength_gago',  # 44.142
        'stock.ord',  # 53.455
        'auto.data.ord_chu',  # 10.475
        'bostonhousing.ord_chu',  # 33.853
        'californiahousing_gago',  # 915.165
        'winequality-red_gago',  # 87.078
        'winequality-white_gago_rev',  #244.255
        'skill_gago',  # 538.764
        'SkillCraft1_rev_7clases',  # 424.998
        'kinematics_gago',  # 84.0114
        'SkillCraft1_rev_8clases',  # 398.771
        'ERA',  # 12.290
        'ailerons_gago',   # 1.916.322
        'abalone.ord_chu',  # 210.760
    ]
    if is_test_run:
        print("WARNING: This is a test run; results are not meaningful")
        estimator_grid = {
            'n_estimators': [10],
            'max_depth': [1, 5],
            'min_samples_leaf': [1],
        }
        n_bags = 3
        n_reps = 1
        n_folds = 2
        dataset_names = [ "ESL" ]
    config = { # store all parameters in a dict
        "seed": seed,
        "n_bags": n_bags,
        "n_reps": n_reps,
        "n_folds": n_folds,
        "option": option,
        "decomposer": decomposer,
        "n_jobs": n_jobs,
        "output_dir": _output_dir(), # fix the directory name
        "estimator": estimator,
        "estimator_grid": estimator_grid,
        "methods": methods,
        "dataset_names": dataset_names
    }
    Parallel(n_jobs=n_jobs)(
        delayed(_repetition_dataset)(i+1, d, config)
        for i in range(n_reps)
        for d in config["dataset_names"]
    )
    emds = _collect_results(config)
    print(emds)

def _output_dir():
    t = datetime.now()  # hour, minute, year, day, month
    tpo = f"{t.year}-{t.month:02d}-{t.day:02d}_{t.hour:02d}:{t.minute:02d}"
    output_dir = 'results/' + tpo
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    return output_dir

def _repetition_dataset(i_rep, dataset_name, config):
    current_seed = config["seed"] + i_rep - 1
    X_trn, X_tst, y_trn, y_tst = _load_data(dataset_name, current_seed)
    print(f"*** Training over {dataset_name}, rep {i_rep}")

    # classifiers are fitted by each object (all methods will use exactly the same predictions)
    # but they checked whether the estimator is already fitted (by a previous object) or not

    # estimator for estimating the testing distribution, GridSearchCV
    estimator = factory.estimator(
        X_trn,
        y_trn,
        config["estimator"],
        config["estimator_grid"],
        current_seed,
        config["n_jobs"],
    )
    estimator_args = {
        "decomposer": {
            "Monotone": factory.Decomposer.monotone,
            "FHTree": factory.Decomposer.fh_tree,
        }[config['decomposer']],
        "option": {
            "CV(DECOMP)": factory.Option.cv_decomp,
            "Bagging(DECOMP)": factory.Option.bagging_decomp,
        }[config['option']],
        "n_folds": np.min([
            config["n_folds"],
            np.min(np.unique(y_trn, return_counts=True)[1])
        ]),
        "random_state": config["seed"],
    }

    print(f"* Training {dataset_name} with {estimator_args['decomposer']} rep {i_rep}")
    methods = {}
    for method_name in config["methods"]:
        if method_name == 'AC':
            m = factory.AC(estimator, **estimator_args)
        if method_name == 'AC_HD':
            m = factory.AC(estimator, distance='HD', **estimator_args)
        if method_name == 'AC_L1':
            m = factory.AC(estimator, distance='L1', **estimator_args)
        if method_name == 'AC_L2':
            m = factory.AC(estimator, distance='L2', **estimator_args)
        if method_name == 'AC_Ord':
            m = factory.OrdinalAC(estimator, **estimator_args)
        if method_name == 'CC':
            m = factory.CC(estimator, **estimator_args)
        if method_name == 'CvMy_Eu':
            m = factory.CvMy(estimator, distances=euclidean_distances, **estimator_args)
        if method_name == 'EDX':
            m = factory.EDX()
        if method_name == 'EDy_EMD':
            m = factory.EDy(estimator, distances=emd_distances, **estimator_args)
        if method_name == 'EDy_Eu':
            m = factory.EDy(estimator, distances=euclidean_distances, **estimator_args)
        if method_name == 'EDy_Ma':
            m = factory.EDy(estimator, **estimator_args)
        if method_name == 'HDX':
            m = factory.HDX(n_bins=BINS_HDX)
        if method_name == 'HDy':
            m = factory.HDy(estimator, n_bins=BINS_GEN, **estimator_args)
        if method_name == 'PAC':
            m = factory.PAC(estimator, **estimator_args)
        if method_name == 'PAC_HD':
            m = factory.PAC(estimator, distance='HD', **estimator_args)
        if method_name == 'PAC_L1':
            m = factory.PAC(estimator, distance='L1', **estimator_args)
        if method_name == 'PAC_L2':
            m = factory.PAC(estimator, distance='L2', **estimator_args)
        if method_name == 'PCC':
            m = factory.PCC(estimator, **estimator_args)
        if method_name == 'PDF_EMD':
            m = factory.PDF(estimator, distance='EMD', n_bins=BINS_PDF_EMD, **estimator_args)
        if method_name == 'PDF_HD':
            m = factory.PDF(estimator, distance='HD', n_bins=BINS_GEN, **estimator_args)
        if method_name == 'PDF_L2':
            m = factory.PDF(estimator, distance='L2', n_bins=BINS_PDF_L2, **estimator_args)
        if callable(method_name): # support for constructors
            method_name, m = method_name(estimator, **estimator_args)
        m.fit(X_trn, y_trn)
        methods[method_name] = m

    df = pd.DataFrame(columns=COLUMNS)
    for i_bag, (X_tst_, y_tst_, prev_true) in enumerate(create_bags_with_multiple_prevalence(
            X_tst,
            y_tst,
            config["n_bags"],
            current_seed
            )):
        for method_name, method in methods.items():
            prev_pred = method.predict(X_tst_)
            df = pd.concat((df, pd.DataFrame([[
                dataset_name,
                method_name,
                config["decomposer"],
                str(i_rep) + 'x' + str(i_bag), # rb
                prev_true,
                prev_pred,
                mean_absolute_error(prev_true, prev_pred),
                mean_squared_error(prev_true, prev_pred),
                emd(prev_true, prev_pred),
                emd_score(prev_true, prev_pred)
            ]], columns=COLUMNS)))
    output_path = "_".join([
        f"{config['output_dir']}/results",
        str(config["option"]),
        f"{config['n_reps']}x{config['n_bags']}CV{config['n_folds']}",
        config["decomposer"],
        dataset_name
    ]) + ".csv"
    df.to_csv(output_path, mode='a', index=None)

def _load_data(dataset_name, current_seed):
    df = pd.read_csv(f"datasets/ordinal/{dataset_name}.csv", sep=';', header=0)
    x = df.iloc[:, :-1].values.astype(np.float64)
    y = df.iloc[:, -1].values.astype(np.int64)
    return train_test_split(x, y, test_size=0.3, random_state=current_seed, stratify=y)

def _collect_results(config):
    all_files = glob.glob(f"{config['output_dir']}/results*.csv") # concatenate all files
    all_files.sort()
    print('**** Processing', len(all_files), 'datasets')
    good_cols = ['dataset', 'method', 'decomposer', 'mae', 'mse', 'emd', 'emd_score']
    ll = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, usecols=good_cols, header=0)
        df = df[df['dataset'] != 'dataset']  # removing lines with duplicated columns
        ll.append(df)
        print('* Reading', filename, len(df), 'bags')

    res_df = pd.concat(ll, axis=0, ignore_index=True)
    res_df = res_df.astype(dtype = {  # convert errors to float
        'mae': 'float',
        'mse': 'float',
        'emd': 'float',
        'emd_score': 'float'
    }).sort_values(by=['dataset'])
    fout = f"{config['output_dir']}/means_{config['option']}_{config['n_reps']}x{config['n_bags']}CV{config['n_folds']}_{len(all_files)}.csv"
    for i_error, error in enumerate(['emd', 'emd_score', 'mae', 'mse']):
        means_df = res_df.groupby(['decomposer', 'dataset', 'method'])[[error]].agg(['mean']).unstack().round(5)
        means_df.columns = config["methods"]
        means_df['error'] = error  # adding a column at the end
        if i_error == 0:
            means_df.to_csv(fout, mode='w')
        else:
            means_df.to_csv(fout, mode='a', header=False)
    return pd.read_csv(fout)

# command line interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Experiment with standard data sets.")
    parser.add_argument(
        "--seed",
        type = int,
        default = 2032,
        metavar = "N",
        help = "random number generator seed (default: 2032)",
    )
    parser.add_argument(
        "--decomposer",
        type = str,
        default = "Monotone",
        help = "\"Monotone\" (default) or \"FHTree\"",
    )
    parser.add_argument(
        "--n_jobs",
        type = int,
        default = -1,
        metavar = "N",
        help = "number of parallel processes (default: -1 = all cores)",
    )
    parser.add_argument(
        "--is_test_run",
        action = "store_true",
        help = "whether to make a test run of this experiment"
    )
    args = parser.parse_args()
    main(**vars(args))
