import sys, os

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

import unittest
import pandas as pd

from get_ogs import count_ogs


df_test = pd.DataFrame({'#query': ['2', '1', '3'], 'eggNOG_OGs':['a@1,b@2', 'b@1,a@2,c@1', 'c@1']})

dict_count = {"b":1, "a":1, "c":2}
dict_id = {"b":['2'], "a":['1'], "c":['1','3']}

class TestCountOGs(unittest.TestCase):
    def test_input(self):
        # test that input is data frame
        self.assertEqual(count_ogs(df_test)[0], dict_count)
        self.assertEqual(count_ogs(df_test)[1], dict_id)
