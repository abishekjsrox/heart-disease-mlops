from heartrisk import data, eda


def test_eda_writes_figures_and_summary(raw_df, tmp_path):
    info = eda.run(data.clean(raw_df), out_dir=tmp_path / "figures")
    assert len(info["figures"]) == 5
    assert all((tmp_path / "figures" / f).exists() for f in info["figures"])
    assert (tmp_path / "eda_summary.json").exists()
