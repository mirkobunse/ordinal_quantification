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

def main(
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
    methods = [
        'AC_L2',
        'AC_Ord',
        'CC',
        'CvMy_Eu',
        'EDX',
        'EDy_EMD',
        'EDy_Eu',
        'HDX',
        'HDy',
        'PAC_L2',
        'PCC',
        'PDF_EMD',
        'PDF_L2',
    ]
    dataset_names = [
        'auto.data.ord_chu',  # 10.475
        'ERA',  # 12.290
        'ESL',  # 5.395  -->5 clases: 3,4,5,6,7
        'bostonhousing.ord_chu',  # 33.853
        'cement_strength_gago',  # 44.142
        'kinematics_gago',  # 84.0114
        'abalone.ord_chu',  # 210.760
        'californiahousing_gago',  # 915.165
        'ailerons_gago',   # 1.916.322
        'LEV',  #11.023
        'stock.ord',  # 53.455
        'SWD',  #23.048
        'winequality-red_gago',  # 87.078
        'winequality-white_gago_rev',  #244.255
        'SkillCraft1_rev_7clases',  # 424.998
        'SkillCraft1_rev_8clases',  # 398.771
        'skill_gago',  # 538.764
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
        methods = [ "AC_L2", "AC_Ord" ]
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

    n_folds = np.min([config["n_folds"], np.min(np.unique(y_trn, return_counts=True)[1])])
    print(f"* Training {dataset_name} with {config['decomposer']} rep {i_rep}")
    if 'AC' in config["methods"]:
        ac = factory.AC(estimator, n_folds=n_folds)
        ac.fit(X_trn, y_trn)
    if 'AC_HD' in config["methods"]:
        ac_hd = factory.AC(estimator, n_folds=n_folds, distance='HD')
        ac_hd.fit(X_trn, y_trn)
    if 'AC_L1' in config["methods"]:
        ac_l1 = factory.AC(estimator, n_folds=n_folds, distance='L1')
        ac_l1.fit(X_trn, y_trn)
    if 'AC_L2' in config["methods"]:
        ac_l2 = factory.AC(estimator, n_folds=n_folds, distance='L2')
        ac_l2.fit(X_trn, y_trn)
    if 'AC_Ord' in config["methods"]:
        ac_ord = factory.OrdinalAC(estimator, n_folds=n_folds)
        ac_ord.fit(X_trn, y_trn)
    if 'CC' in config["methods"]:
        cc = factory.CC(estimator, n_folds=n_folds)
        cc.fit(X_trn, y_trn)
    if 'CvMy_Eu' in config["methods"]:
        cvmy_eu = factory.CvMy(estimator, n_folds=n_folds, distances=euclidean_distances)
        cvmy_eu.fit(X_trn, y_trn)
    if 'EDX' in config["methods"]:
        edx = factory.EDX()
        edx.fit(X_trn, y_trn)
    if 'EDy_EMD' in config["methods"]:
        edy_emd = factory.EDy(estimator, n_folds=n_folds, distances=emd_distances)
        edy_emd.fit(X_trn, y_trn)
    if 'EDy_Eu' in config["methods"]:
        edy_eu = factory.EDy(estimator, n_folds=n_folds, distances=euclidean_distances)
        edy_eu.fit(X_trn, y_trn)
    if 'EDy_Ma' in config["methods"]:
        edy_ma = factory.EDy(estimator, n_folds=n_folds)
        edy_ma.fit(X_trn, y_trn)
    if 'HDX' in config["methods"]:
        hdx = factory.HDX(n_bins=BINS_HDX)
        hdx.fit(X_trn, y_trn)
    if 'HDy' in config["methods"]:
        hdy = factory.HDy(estimator, n_folds=n_folds, n_bins=BINS_GEN)
        hdy.fit(X_trn, y_trn)
    if 'PAC' in config["methods"]:
        pac = factory.PAC(estimator, n_folds=n_folds)
        pac.fit(X_trn, y_trn)
    if 'PAC_HD' in config["methods"]:
        pac_hd = factory.PAC(estimator, n_folds=n_folds, distance='HD')
        pac_hd.fit(X_trn, y_trn)
    if 'PAC_L1' in config["methods"]:
        pac_l1 = factory.PAC(estimator, n_folds=n_folds, distance='L1')
        pac_l1.fit(X_trn, y_trn)
    if 'PAC_L2' in config["methods"]:
        pac_l2 = factory.PAC(estimator, n_folds=n_folds, distance='L2')
        pac_l2.fit(X_trn, y_trn)
    if 'PCC' in config["methods"]:
        pcc = factory.PCC(estimator, n_folds=n_folds)
        pcc.fit(X_trn, y_trn)
    if 'PDF_EMD' in config["methods"]:
        pdf_emd = factory.PDF(estimator, n_folds=n_folds, distance='EMD', n_bins=BINS_PDF_EMD)
        pdf_emd.fit(X_trn, y_trn)
    if 'PDF_HD' in config["methods"]:
        pdf_hd = factory.PDF(estimator, n_folds=n_folds, distance='HD', n_bins=BINS_GEN)
        pdf_hd.fit(X_trn, y_trn)
    if 'PDF_L2' in config["methods"]:
        pdf_l2 = factory.PDF(estimator, n_folds=n_folds, distance='L2', n_bins=BINS_PDF_L2)
        pdf_l2.fit(X_trn, y_trn)

    df = pd.DataFrame(columns=COLUMNS)
    for i_bag, (X_tst_, y_tst_, prev_true) in enumerate(create_bags_with_multiple_prevalence(
            X_tst,
            y_tst,
            config["n_bags"],
            current_seed
            )):
        prev_preds = []
        if 'AC' in config["methods"]:
            prev_preds.append(ac.predict(X_tst_))
        if 'AC_HD' in config["methods"]:
            prev_preds.append(ac_hd.predict(X_tst_))
        if 'AC_L1' in config["methods"]:
            prev_preds.append(ac_l1.predict(X_tst_))
        if 'AC_L2' in config["methods"]:
            prev_preds.append(ac_l2.predict(X_tst_))
        if 'AC_Ord' in config["methods"]:
            prev_preds.append(ac_ord.predict(X_tst_))
        if 'CC' in config["methods"]:
            prev_preds.append(cc.predict(X_tst_))
        if 'CvMy_Eu' in config["methods"]:
            prev_preds.append(cvmy_eu.predict(X_tst_))
        if 'EDX' in config["methods"]:
            prev_preds.append(edx.predict(X_tst_))
        if 'EDy_EMD' in config["methods"]:
            prev_preds.append(edy_emd.predict(X_tst_))
        if 'EDy_Eu' in config["methods"]:
            prev_preds.append(edy_eu.predict(X_tst_))
        if 'EDy_Ma' in config["methods"]:
            prev_preds.append(edy_ma.predict(X_tst_))
        if 'HDX' in config["methods"]:
            prev_preds.append(hdx.predict(X_tst_))
        if 'HDy' in config["methods"]:
            prev_preds.append(hdy.predict(X_tst_))
        if 'PAC' in config["methods"]:
            prev_preds.append(pac.predict(X_tst_))
        if 'PAC_HD' in config["methods"]:
            prev_preds.append(pac_hd.predict(X_tst_))
        if 'PAC_L1' in config["methods"]:
            prev_preds.append(pac_l1.predict(X_tst_))
        if 'PAC_L2' in config["methods"]:
            prev_preds.append(pac_l2.predict(X_tst_))
        if 'PCC' in config["methods"]:
            prev_preds.append(pcc.predict(X_tst_))
        if 'PDF_EMD' in config["methods"]:
            prev_preds.append(pdf_emd.predict(X_tst_))
        if 'PDF_HD' in config["methods"]:
            prev_preds.append(pdf_hd.predict(X_tst_))
        if 'PDF_L2' in config["methods"]:
            prev_preds.append(pdf_l2.predict(X_tst_))
        for n_method, (method, prev_pred) in enumerate(zip(config["methods"], prev_preds)):
            df = pd.concat((df, pd.DataFrame([[
                dataset_name,
                method,
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
