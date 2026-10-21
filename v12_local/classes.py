"""Class names and the Group A / Group B split, defined once for every script.

Group B (lesion) holds the classes whose median box covers less than half the image; the rest are
Group A (whole-leaf). analyse_protocol.py measures this on the labels and asserts that GROUP_B
below still matches, so a stale list fails loudly instead of quietly skewing the A/B numbers.

GROUP_B changed on 2026-10-21: potato_healthy moved to Group A. Its polygon labels, read as boxes,
had made its boxes look lesion-sized; read correctly, its median box covers 0.67 of the image (v13).
Per-class CSVs, summary JSONs and eval JSONs written before that date carry the old grouping in
their `group`, `A`/`B` and `A_whole_leaf`/`B_lesion` fields. Recompute group means from the
per-class values with `group_of` instead of trusting those stored fields.
"""
NAMES = [
    "bean_angular_leaf_spot", "bean_bean_rust", "bean_healthy",
    "cauliflower_bacterial_spot_rot", "cauliflower_black_rot",
    "cauliflower_downy_mildew", "cauliflower_no_disease",
    "chilli_Curling-Leafs", "chilli_Healthy-Leafs", "chilli_Leaf Spot-Leafs",
    "chilli_Leveillula taurica-Leafs", "chilli_Mozaik Leaf",
    "cucumber_Antraknosa", "cucumber_Bacterial Wilt", "cucumber_Downy Mildew",
    "potato_healthy", "potato_blackleg", "potato_late_blight", "potato_early_blight",
    "tomato_bacterial_spot", "tomato_curl_virus", "tomato_two_spotted_spider_mite",
]
GROUP_B = {
    "bean_angular_leaf_spot", "bean_bean_rust", "cucumber_Antraknosa", "cucumber_Bacterial Wilt",
    "potato_blackleg", "potato_early_blight", "potato_late_blight",
}
GROUP_B_V12_PAPER = GROUP_B | {"potato_healthy"}   # the published list, for tracing old outputs


def group_of(name):
    return "B_lesion" if name in GROUP_B else "A_whole_leaf"
