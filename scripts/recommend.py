"""Illustrative, rule-based follow-up notes; not learned treatment advice."""

RECS = {
    "Tomato_Late_blight": "Inspect for spreading lesions; seek local plant-health guidance before treatment.",
    "Tomato_Early_blight": "Review leaf wetness, crop rotation, and removal of affected debris.",
    "Tomato_Leaf_Mold": "Review humidity, airflow, and sanitation; confirm the diagnosis locally.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Inspect for whiteflies and confirm suspected viral symptoms.",
    "Tomato__Tomato_mosaic_virus": "Review tool hygiene and handling practices; confirm suspected viral symptoms.",
    "Tomato_Bacterial_spot": "Review leaf wetness and sanitation; seek local confirmation.",
    "Tomato_Septoria_leaf_spot": "Inspect lower leaves and review irrigation and debris management.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Inspect leaf undersides for mites and consult local pest-management guidance.",
    "Tomato__Target_Spot": "Review airflow and leaf wetness; confirm lesions with local guidance.",
    "Pepper__bell___Bacterial_spot": "Review leaf wetness, sanitation, and rotation; confirm symptoms locally.",
    "Potato___Early_blight": "Review crop nutrition and debris management; confirm symptoms locally.",
    "Potato___Late_blight": "Inspect for rapid spread and consult local plant-health guidance promptly.",
    "Tomato_healthy": "Predicted healthy class; continue routine visual scouting.",
    "Pepper__bell___healthy": "Predicted healthy class; continue routine visual scouting.",
    "Potato___healthy": "Predicted healthy class; continue routine visual scouting.",
}
DEFAULT = "Inspect the plant and consult local plant-health guidance if symptoms persist."


def recommend_for(label: str) -> str:
    """Return a static note for a model label, with a fallback for unknown labels."""
    return RECS.get(label, DEFAULT)


if __name__ == "__main__":
    for label in sorted(RECS):
        print(f"{label} -> {recommend_for(label)}")
