import numpy as np

from heartrisk import config, data

UCI_TEXT = ("63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0\n"
            "67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2\n"
            "53.0,0.0,3.0,128.0,216.0,0.0,2.0,115.0,0.0,0.0,1.0,0.0,?,0\n")

MIRROR_TEXT = (",Age,Sex,ChestPain,RestBP,Chol,Fbs,RestECG,MaxHR,ExAng,Oldpeak,Slope,Ca,Thal,AHD\n"
               "1,63,1,typical,145,233,1,2,150,0,2.3,3,0,fixed,No\n"
               "2,67,1,asymptomatic,160,286,0,2,108,1,1.5,2,3,normal,Yes\n")


def test_parse_uci_handles_question_marks():
    df = data.parse_uci(UCI_TEXT)
    assert list(df.columns) == config.UCI_COLUMNS
    assert df["thal"].isna().sum() == 1


def test_parse_mirror_maps_to_uci_codes():
    df = data.parse_mirror(MIRROR_TEXT)
    assert list(df.columns) == config.UCI_COLUMNS
    assert df["cp"].tolist() == [1, 4]
    assert df["thal"].tolist() == [6, 3]
    assert df["num"].tolist() == [0, 1]


def test_clean_binarises_target_and_drops_num(raw_df):
    out = data.clean(raw_df)
    assert "num" not in out.columns
    assert set(out[config.TARGET].unique()) <= {0, 1}
    assert (out[config.TARGET] == (raw_df["num"] > 0).astype(int)).all()


def test_clean_turns_invalid_values_into_missing(raw_df):
    out = data.clean(raw_df)
    assert np.isnan(out.loc[1, "thal"])
    assert np.isnan(out.loc[2, "chol"])
    assert np.isnan(out.loc[0, "ca"])


def test_clean_removes_duplicates(raw_df):
    doubled = raw_df._append(raw_df.iloc[[5]], ignore_index=True)
    assert len(data.clean(doubled)) == len(data.clean(raw_df))
