"""任务类型、类别、special tokens 常量。"""

# 任务类型
TASK_FACILITY = "facility"
TASK_REGION = "region"
TASK_DISASTER = "disaster"

# Special tokens
SEG_TOKEN = "<SEG#1>"
EVIDENCE_START = "<EVIDENCE>"
EVIDENCE_END = "</EVIDENCE>"

# 槽位标签（解释三段）
FUNCTION_TAG = "[function]"
STRUCTURE_TAG = "[structure]"
CONTEXT_TAG = "[context]"

# 掩膜/ignore 标签
BACKGROUND_ID = 0
IGNORE_INDEX = 255

# 关键词（用于奖励/正则）——可根据数据进一步补充/替换
KEYWORDS_STRUCTURE = [
    "rectangular",
    "circular",
    "cylindrical",
    "linear",
    "grid",
    "dense",
    "sparse",
    "pitch",
    "goal",
    "lane",
    "runway",
    "bridge",
    "dock",
    "terminal",
    "storage tank",
    "field",
]
KEYWORDS_CONTEXT = [
    "near",
    "adjacent",
    "surrounded",
    "along",
    "road",
    "highway",
    "river",
    "coast",
    "sea",
    "open",
    "park",
    "residential",
    "industrial",
    "commercial",
    "vegetation",
    "forest",
]

# 功能关键词（供 function 槽位相似度/关键词命中使用，可按数据集扩充）
KEYWORDS_FUNCTION = [
    "function",
    "purpose",
    "use",
    "utility",
    "suitable",
    "fit",
    "serve",
    "support",
    "practice",
    "training",
    "facility",
    "area",
    "space",
]

# 类别映射（示例，按需补全你的类别表）
CLASS_ID_TO_NAME = {
    0: "background",
    1: "Harbor/Port",
    2: "Airport",
    3: "Parking Lot",
    4: "Baseball Field",
    5: "Basketball Court",
    6: "Bridge",
    7: "Dam",
    8: "Storage Tank",
    9: "Tennis Court",
    10: "Road",
    11: "Runway",
    12: "Shipyard",
    13: "Construction Site",
    14: "Railway",
    15: "Soccer Field",
    16: "Industrial Plant",
    17: "Residential Area",
    18: "Commercial Area",
    19: "Stadium",
    20: "Other",
}
