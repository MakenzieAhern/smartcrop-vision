from scripts.labels import load_labels
from scripts.recommend import DEFAULT, RECS, recommend_for


def test_all_model_labels_have_explicit_notes():
    assert set(load_labels()) == set(RECS)
    assert all(recommend_for(label) != DEFAULT for label in load_labels())


def test_unknown_label_returns_general_note():
    assert recommend_for("unseen_disease") == DEFAULT
