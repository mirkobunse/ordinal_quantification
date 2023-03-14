import tempfile
import numpy as np
import pandas as pd
from unittest import TestCase

from ordinal_quantification.experiments.__main__ import main

RNG = np.random.RandomState(876) # make tests reproducible

class TestExperiments(TestCase):
  def test_reproducibility(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
        main_args = {
            "is_test_run": True,
            "output_dir": tmp_dir,
        }
        df_ref = pd.read_csv(main(**main_args)) # reference results

        df_tst = pd.read_csv(main(**main_args)) # same config, same seed
        self.assertTrue(df_tst.equals(df_ref))

        df_tst = pd.read_csv(main(n_jobs=1, **main_args)) # n_jobs != -1
        self.assertTrue(df_tst.equals(df_ref))

        # FIXME results differ when other methods are selected:
        #
        # methods = [ "EDy_Eu", "EDy_EMD", "PDF_EMD" ]
        # columns = ["decomposer", "dataset"] + methods + ["error"]
        # df_tst = pd.read_csv(main(methods=methods, **main_args))
        # self.assertTrue(df_tst[columns].equals(df_ref[columns]))
