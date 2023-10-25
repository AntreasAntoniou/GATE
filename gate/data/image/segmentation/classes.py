ade20_classes = (
    "wall",
    "building",
    "sky",
    "floor",
    "tree",
    "ceiling",
    "road",
    "bed ",
    "windowpane",
    "grass",
    "cabinet",
    "sidewalk",
    "person",
    "earth",
    "door",
    "table",
    "mountain",
    "plant",
    "curtain",
    "chair",
    "car",
    "water",
    "painting",
    "sofa",
    "shelf",
    "house",
    "sea",
    "mirror",
    "rug",
    "field",
    "armchair",
    "seat",
    "fence",
    "desk",
    "rock",
    "wardrobe",
    "lamp",
    "bathtub",
    "railing",
    "cushion",
    "base",
    "box",
    "column",
    "signboard",
    "chest of drawers",
    "counter",
    "sand",
    "sink",
    "skyscraper",
    "fireplace",
    "refrigerator",
    "grandstand",
    "path",
    "stairs",
    "runway",
    "case",
    "pool table",
    "pillow",
    "screen door",
    "stairway",
    "river",
    "bridge",
    "bookcase",
    "blind",
    "coffee table",
    "toilet",
    "flower",
    "book",
    "hill",
    "bench",
    "countertop",
    "stove",
    "palm",
    "kitchen island",
    "computer",
    "swivel chair",
    "boat",
    "bar",
    "arcade machine",
    "hovel",
    "bus",
    "towel",
    "light",
    "truck",
    "tower",
    "chandelier",
    "awning",
    "streetlight",
    "booth",
    "television receiver",
    "airplane",
    "dirt track",
    "apparel",
    "pole",
    "land",
    "bannister",
    "escalator",
    "ottoman",
    "bottle",
    "buffet",
    "poster",
    "stage",
    "van",
    "ship",
    "fountain",
    "conveyer belt",
    "canopy",
    "washer",
    "plaything",
    "swimming pool",
    "stool",
    "barrel",
    "basket",
    "waterfall",
    "tent",
    "bag",
    "minibike",
    "cradle",
    "oven",
    "ball",
    "food",
    "step",
    "tank",
    "trade name",
    "microwave",
    "pot",
    "animal",
    "bicycle",
    "lake",
    "dishwasher",
    "screen",
    "blanket",
    "sculpture",
    "hood",
    "sconce",
    "vase",
    "traffic light",
    "tray",
    "ashcan",
    "fan",
    "pier",
    "crt screen",
    "plate",
    "monitor",
    "bulletin board",
    "shower",
    "radiator",
    "glass",
    "clock",
    "flag",
)

cityscapes_classes = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)

# In segmentation map annotation for COCO-Stuff, Train-IDs of the 10k version
# are from 1 to 171, where 0 is the ignore index, and Train-ID of COCO Stuff
# 164k is from 0 to 170, where 255 is the ignore index.

cocostuff_classes = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
    "banner",
    "blanket",
    "branch",
    "bridge",
    "building-other",
    "bush",
    "cabinet",
    "cage",
    "cardboard",
    "carpet",
    "ceiling-other",
    "ceiling-tile",
    "cloth",
    "clothes",
    "clouds",
    "counter",
    "cupboard",
    "curtain",
    "desk-stuff",
    "dirt",
    "door-stuff",
    "fence",
    "floor-marble",
    "floor-other",
    "floor-stone",
    "floor-tile",
    "floor-wood",
    "flower",
    "fog",
    "food-other",
    "fruit",
    "furniture-other",
    "grass",
    "gravel",
    "ground-other",
    "hill",
    "house",
    "leaves",
    "light",
    "mat",
    "metal",
    "mirror-stuff",
    "moss",
    "mountain",
    "mud",
    "napkin",
    "net",
    "paper",
    "pavement",
    "pillow",
    "plant-other",
    "plastic",
    "platform",
    "playingfield",
    "railing",
    "railroad",
    "river",
    "road",
    "rock",
    "roof",
    "rug",
    "salad",
    "sand",
    "sea",
    "shelf",
    "sky-other",
    "skyscraper",
    "snow",
    "solid-other",
    "stairs",
    "stone",
    "straw",
    "structural-other",
    "table",
    "tent",
    "textile-other",
    "towel",
    "tree",
    "vegetable",
    "wall-brick",
    "wall-concrete",
    "wall-other",
    "wall-panel",
    "wall-stone",
    "wall-tile",
    "wall-wood",
    "water-other",
    "waterdrops",
    "window-blind",
    "window-other",
    "wood",
)

# In segmentation map annotation for COCO-Stuff, Train-IDs of the 10k version
# are from 1 to 171, where 0 is the ignore index, and Train-ID of COCO Stuff
# 164k is from 0 to 170, where 255 is the ignore index.

cocostuff_10K_classes = cocostuff_classes[1:]
cocostuff_164k_classes = cocostuff_classes[:-1]

nyu_depth_v2_classes = (
    "bed",
    "objects",
    "chair",
    "furniture",
    "ceiling",
    "floor",
    "wall",
    "window",
    "building",
    "sky",
    "cabinet",
    "table",
    "door",
    "light",
    "sofa",
    "shelf",
    "stairs",
    "curtain",
    "dresser",
    "pillow",
    "mirror",
    "floormat",
    "clothes",
    "books",
    "refrigerator",
    "television",
    "paper",
    "towel",
    "shower curtain",
    "box",
    "whiteboard",
    "person",
    "night stand",
    "toilet",
    "sink",
    "lamp",
    "bathtub",
    "bag",
)


pascal_context_classes = (
    "background",
    "aeroplane",
    "bag",
    "bed",
    "bedclothes",
    "bench",
    "bicycle",
    "bird",
    "boat",
    "book",
    "bottle",
    "building",
    "bus",
    "cabinet",
    "car",
    "cat",
    "ceiling",
    "chair",
    "cloth",
    "computer",
    "cow",
    "cup",
    "curtain",
    "dog",
    "door",
    "fence",
    "floor",
    "flower",
    "food",
    "grass",
    "ground",
    "horse",
    "keyboard",
    "light",
    "motorbike",
    "mountain",
    "mouse",
    "person",
    "plate",
    "platform",
    "pottedplant",
    "road",
    "rock",
    "sheep",
    "shelves",
    "sidewalk",
    "sign",
    "sky",
    "snow",
    "sofa",
    "table",
    "track",
    "train",
    "tree",
    "truck",
    "tvmonitor",
    "wall",
    "water",
    "window",
    "wood",
)

medical_decathlon_labels = {
    "task01braintumour": [
        "Background",
        "Necrotic and Non-Enhancing Tumor",
        "Edema",
        "Enhancing Tumor",
    ],
    "task02heart": ["Background", "Left Atrium"],
    "task03liver": ["Background", "Liver", "Tumour"],
    "task04hippocampus": ["Background", "Anterior", "Posterior"],
    "task05prostate": [
        "Background",
        "Peripheral Zone",
        "Transitional Zone",
        "Central Gland",
    ],
    "task06lung": ["Background", "Lung", "Tumour"],
    "task07pancreas": ["Background", "Pancreas", "Tumour"],
    "task08hepaticvessel": ["Background", "Vessels", "Tumour"],
    "task09spleen": ["Background", "Spleen"],
    "task10colon": ["Background", "Tumour"],
}

acdc_labels = ["Background", "Left Ventricle", "Right Ventricle", "Myocardium"]
