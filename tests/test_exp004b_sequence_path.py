"""EXP004-B path errors must refer to the exact EXP004-A input sequence."""
import json
from argparse import Namespace
import numpy as np
import pandas as pd
import pytest
from vio_iqa.exp004b import evaluate

def test_missing_sequence_reports_original_exp004a_root(tmp_path):
    old=tmp_path/"previous_success"
    old.mkdir()
    expected=tmp_path/"real_EuRoC"/"MH_01_easy"
    (old/"manifest.json").write_text(json.dumps({"sequence":str(expected)}))
    wrong=tmp_path/"incorrect"/"MH_01_easy"
    with pytest.raises(FileNotFoundError,match="EXP004-A successfully used") as exc:
        evaluate(Namespace(sequence=wrong,exp004_dir=old,output_dir=tmp_path/"results"))
    assert str(expected) in str(exc.value)
    assert str(wrong/"mav0"/"cam0"/"data.csv") in str(exc.value)
    assert str(wrong/"cam0"/"data.csv") in str(exc.value)
