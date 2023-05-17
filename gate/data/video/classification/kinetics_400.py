"""
Download and extract kinetics-400 dataset.
Original code and hosting from https://github.com/cvdfoundation/kinetics-dataset, but re-written in Python.
"""
import logging
import shutil
import tarfile
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# KINETICS-400
"""     videos      csv         replace
test:   38685       39805       0
train:  240258      246534      1392
val:    19881       19906       0
"""


SPLITS = ["test", "train", "val"]
NUM_TRAIN_PARTS = 242
NUM_VAL_PARTS = 20
NUM_TEST_PARTS = 39

DOWNLOAD_GB = 450  # more precisely, 435 GiB
EXTRACT_GB = 450  # more precisely, 438 GiB

NUM_FILES_EACH_CLASS = {}
NUM_FILES_EACH_CLASS["train"] = {
    "abseiling": 988,
    "air_drumming": 977,
    "answering_questions": 321,
    "applauding": 256,
    "applying_cream": 318,
    "archery": 978,
    "arm_wrestling": 954,
    "arranging_flowers": 424,
    "assembling_computer": 376,
    "auctioning": 326,
    "baby_waking_up": 451,
    "baking_cookies": 737,
    "balloon_blowing": 651,
    "bandaging": 413,
    "barbequing": 902,
    "bartending": 440,
    "beatboxing": 782,
    "bee_keeping": 279,
    "belly_dancing": 908,
    "bench_pressing": 948,
    "bending_back": 472,
    "bending_metal": 255,
    "biking_through_snow": 895,
    "blasting_sand": 558,
    "blowing_glass": 979,
    "blowing_leaves": 252,
    "blowing_nose": 420,
    "blowing_out_candles": 989,
    "bobsledding": 400,
    "bookbinding": 758,
    "bouncing_on_trampoline": 529,
    "bowling": 917,
    "braiding_hair": 573,
    "breading_or_breadcrumbing": 299,
    "breakdancing": 789,
    "brush_painting": 376,
    "brushing_hair": 761,
    "brushing_teeth": 982,
    "building_cabinet": 279,
    "building_shed": 275,
    "bungee_jumping": 893,
    "busking": 693,
    "canoeing_or_kayaking": 989,
    "capoeira": 940,
    "carrying_baby": 406,
    "cartwheeling": 455,
    "carving_pumpkin": 548,
    "catching_fish": 501,
    "catching_or_throwing_baseball": 599,
    "catching_or_throwing_frisbee": 891,
    "catching_or_throwing_softball": 688,
    "celebrating": 573,
    "changing_oil": 550,
    "changing_wheel": 309,
    "checking_tires": 399,
    "cheerleading": 984,
    "chopping_wood": 756,
    "clapping": 334,
    "clay_pottery_making": 358,
    "clean_and_jerk": 745,
    "cleaning_floor": 703,
    "cleaning_gutters": 445,
    "cleaning_pool": 294,
    "cleaning_shoes": 535,
    "cleaning_toilet": 418,
    "cleaning_windows": 531,
    "climbing_a_rope": 262,
    "climbing_ladder": 506,
    "climbing_tree": 953,
    "contact_juggling": 978,
    "cooking_chicken": 811,
    "cooking_egg": 455,
    "cooking_on_campfire": 250,
    "cooking_sausages": 311,
    "counting_money": 503,
    "country_line_dancing": 858,
    "cracking_neck": 290,
    "crawling_baby": 987,
    "crossing_river": 787,
    "crying": 843,
    "curling_hair": 687,
    "cutting_nails": 404,
    "cutting_pineapple": 548,
    "cutting_watermelon": 600,
    "dancing_ballet": 982,
    "dancing_charleston": 566,
    "dancing_gangnam_style": 678,
    "dancing_macarena": 803,
    "deadlifting": 652,
    "decorating_the_christmas_tree": 436,
    "digging": 250,
    "dining": 520,
    "disc_golfing": 415,
    "diving_cliff": 914,
    "dodgeball": 440,
    "doing_aerobics": 302,
    "doing_laundry": 307,
    "doing_nails": 758,
    "drawing": 291,
    "dribbling_basketball": 762,
    "drinking": 437,
    "drinking_beer": 409,
    "drinking_shots": 253,
    "driving_car": 911,
    "driving_tractor": 748,
    "drop_kicking": 552,
    "drumming_fingers": 257,
    "dunking_basketball": 934,
    "dying_hair": 885,
    "eating_burger": 698,
    "eating_cake": 341,
    "eating_carrots": 361,
    "eating_chips": 579,
    "eating_doughnuts": 373,
    "eating_hotdog": 410,
    "eating_ice_cream": 762,
    "eating_spaghetti": 987,
    "eating_watermelon": 397,
    "egg_hunting": 349,
    "exercising_arm": 265,
    "exercising_with_an_exercise_ball": 288,
    "extinguishing_fire": 441,
    "faceplanting": 287,
    "feeding_birds": 984,
    "feeding_fish": 810,
    "feeding_goats": 864,
    "filling_eyebrows": 892,
    "finger_snapping": 674,
    "fixing_hair": 510,
    "flipping_pancake": 556,
    "flying_kite": 908,
    "folding_clothes": 523,
    "folding_napkins": 693,
    "folding_paper": 718,
    "front_raises": 801,
    "frying_vegetables": 433,
    "garbage_collecting": 291,
    "gargling": 277,
    "getting_a_haircut": 497,
    "getting_a_tattoo": 584,
    "giving_or_receiving_award": 795,
    "golf_chipping": 545,
    "golf_driving": 684,
    "golf_putting": 930,
    "grinding_meat": 263,
    "grooming_dog": 457,
    "grooming_horse": 486,
    "gymnastics_tumbling": 987,
    "hammer_throw": 992,
    "headbanging": 935,
    "headbutting": 479,
    "high_jump": 798,
    "high_kick": 664,
    "hitting_baseball": 909,
    "hockey_stop": 318,
    "holding_snake": 274,
    "hopscotch": 568,
    "hoverboarding": 406,
    "hugging": 355,
    "hula_hooping": 974,
    "hurdling": 467,
    "hurling_(sport)": 677,
    "ice_climbing": 691,
    "ice_fishing": 377,
    "ice_skating": 971,
    "ironing": 383,
    "javelin_throw": 745,
    "jetskiing": 975,
    "jogging": 269,
    "juggling_balls": 766,
    "juggling_fire": 509,
    "juggling_soccer_ball": 327,
    "jumping_into_pool": 965,
    "jumpstyle_dancing": 508,
    "kicking_field_goal": 677,
    "kicking_soccer_ball": 391,
    "kissing": 479,
    "kitesurfing": 642,
    "knitting": 527,
    "krumping": 506,
    "laughing": 735,
    "laying_bricks": 270,
    "long_jump": 678,
    "lunge": 604,
    "making_a_cake": 297,
    "making_a_sandwich": 285,
    "making_bed": 522,
    "making_jewelry": 502,
    "making_pizza": 977,
    "making_snowman": 594,
    "making_sushi": 281,
    "making_tea": 264,
    "marching": 987,
    "massaging_back": 903,
    "massaging_feet": 310,
    "massaging_legs": 362,
    "massaging_person's_head": 484,
    "milking_cow": 811,
    "mopping_floor": 444,
    "motorcycling": 951,
    "moving_furniture": 269,
    "mowing_lawn": 979,
    "news_anchoring": 267,
    "opening_bottle": 576,
    "opening_present": 696,
    "paragliding": 646,
    "parasailing": 608,
    "parkour": 348,
    "passing_American_football_(in_game)": 711,
    "passing_American_football_(not_in_game)": 882,
    "peeling_apples": 431,
    "peeling_potatoes": 297,
    "petting_animal_(not_cat)": 604,
    "petting_cat": 599,
    "picking_fruit": 627,
    "planting_trees": 404,
    "plastering": 272,
    "playing_accordion": 768,
    "playing_badminton": 763,
    "playing_bagpipes": 677,
    "playing_basketball": 987,
    "playing_bass_guitar": 960,
    "playing_cards": 571,
    "playing_cello": 919,
    "playing_chess": 665,
    "playing_clarinet": 861,
    "playing_controller": 365,
    "playing_cricket": 718,
    "playing_cymbals": 484,
    "playing_didgeridoo": 634,
    "playing_drums": 743,
    "playing_flute": 320,
    "playing_guitar": 952,
    "playing_harmonica": 842,
    "playing_harp": 994,
    "playing_ice_hockey": 750,
    "playing_keyboard": 559,
    "playing_kickball": 314,
    "playing_monopoly": 573,
    "playing_organ": 514,
    "playing_paintball": 979,
    "playing_piano": 527,
    "playing_poker": 936,
    "playing_recorder": 980,
    "playing_saxophone": 752,
    "playing_squash_or_racquetball": 816,
    "playing_tennis": 967,
    "playing_trombone": 979,
    "playing_trumpet": 823,
    "playing_ukulele": 986,
    "playing_violin": 977,
    "playing_volleyball": 629,
    "playing_xylophone": 591,
    "pole_vault": 820,
    "presenting_weather_forecast": 888,
    "pull_ups": 954,
    "pumping_fist": 849,
    "pumping_gas": 384,
    "punching_bag": 989,
    "punching_person_(boxing)": 335,
    "push_up": 457,
    "pushing_car": 910,
    "pushing_cart": 990,
    "pushing_wheelchair": 310,
    "reading_book": 971,
    "reading_newspaper": 266,
    "recording_music": 262,
    "riding_a_bike": 321,
    "riding_camel": 557,
    "riding_elephant": 942,
    "riding_mechanical_bull": 539,
    "riding_mountain_bike": 338,
    "riding_mule": 324,
    "riding_or_walking_with_horse": 971,
    "riding_scooter": 514,
    "riding_unicycle": 704,
    "ripping_paper": 447,
    "robot_dancing": 735,
    "rock_climbing": 984,
    "rock_scissors_paper": 271,
    "roller_skating": 801,
    "running_on_treadmill": 271,
    "sailing": 711,
    "salsa_dancing": 986,
    "sanding_floor": 422,
    "scrambling_eggs": 646,
    "scuba_diving": 807,
    "setting_table": 323,
    "shaking_hands": 481,
    "shaking_head": 726,
    "sharpening_knives": 270,
    "sharpening_pencil": 595,
    "shaving_head": 787,
    "shaving_legs": 352,
    "shearing_sheep": 831,
    "shining_shoes": 450,
    "shooting_basketball": 439,
    "shooting_goal_(soccer)": 279,
    "shot_put": 824,
    "shoveling_snow": 719,
    "shredding_paper": 249,
    "shuffling_cards": 657,
    "side_kick": 822,
    "sign_language_interpreting": 288,
    "singing": 980,
    "situp": 653,
    "skateboarding": 956,
    "ski_jumping": 874,
    "skiing_(not_slalom_or_crosscountry)": 971,
    "skiing_crosscountry": 318,
    "skiing_slalom": 352,
    "skipping_rope": 333,
    "skydiving": 350,
    "slacklining": 639,
    "slapping": 304,
    "sled_dog_racing": 623,
    "smoking": 893,
    "smoking_hookah": 686,
    "snatch_weight_lifting": 787,
    "sneezing": 341,
    "sniffing": 245,
    "snorkeling": 853,
    "snowboarding": 787,
    "snowkiting": 990,
    "snowmobiling": 453,
    "somersaulting": 825,
    "spinning_poi": 981,
    "spray_painting": 745,
    "spraying": 317,
    "springboard_diving": 228,
    "squat": 985,
    "sticking_tongue_out": 610,
    "stomping_grapes": 291,
    "stretching_arm": 557,
    "stretching_leg": 660,
    "strumming_guitar": 318,
    "surfing_crowd": 721,
    "surfing_water": 598,
    "sweeping_floor": 440,
    "swimming_backstroke": 912,
    "swimming_breast_stroke": 680,
    "swimming_butterfly_stroke": 521,
    "swing_dancing": 355,
    "swinging_legs": 255,
    "swinging_on_something": 320,
    "sword_fighting": 317,
    "tai_chi": 910,
    "taking_a_shower": 227,
    "tango_dancing": 955,
    "tap_dancing": 786,
    "tapping_guitar": 658,
    "tapping_pen": 539,
    "tasting_beer": 436,
    "tasting_food": 440,
    "testifying": 346,
    "texting": 547,
    "throwing_axe": 662,
    "throwing_ball": 479,
    "throwing_discus": 943,
    "tickling": 442,
    "tobogganing": 989,
    "tossing_coin": 309,
    "tossing_salad": 307,
    "training_dog": 328,
    "trapezing": 634,
    "trimming_or_shaving_beard": 806,
    "trimming_trees": 508,
    "triple_jump": 628,
    "tying_bow_tie": 217,
    "tying_knot_(not_on_a_tie)": 675,
    "tying_tie": 510,
    "unboxing": 694,
    "unloading_truck": 253,
    "using_computer": 757,
    "using_remote_controller_(not_gaming)": 393,
    "using_segway": 221,
    "vault": 410,
    "waiting_in_line": 278,
    "walking_the_dog": 974,
    "washing_dishes": 879,
    "washing_feet": 696,
    "washing_hair": 253,
    "washing_hands": 750,
    "water_skiing": 614,
    "water_sliding": 264,
    "watering_plants": 525,
    "waxing_back": 385,
    "waxing_chest": 605,
    "waxing_eyebrows": 558,
    "waxing_legs": 778,
    "weaving_basket": 588,
    "welding": 601,
    "whistling": 262,
    "windsurfing": 957,
    "wrapping_present": 672,
    "wrestling": 335,
    "writing": 564,
    "yawning": 238,
    "yoga": 947,
    "zumba": 939,
}


NUM_FILES_EACH_CLASS["val"] = {
    "abseiling": 50,
    "air_drumming": 49,
    "answering_questions": 50,
    "applauding": 50,
    "applying_cream": 50,
    "archery": 50,
    "arm_wrestling": 49,
    "arranging_flowers": 50,
    "assembling_computer": 50,
    "auctioning": 50,
    "baby_waking_up": 50,
    "baking_cookies": 49,
    "balloon_blowing": 49,
    "bandaging": 50,
    "barbequing": 50,
    "bartending": 50,
    "beatboxing": 50,
    "bee_keeping": 50,
    "belly_dancing": 49,
    "bench_pressing": 50,
    "bending_back": 50,
    "bending_metal": 50,
    "biking_through_snow": 50,
    "blasting_sand": 50,
    "blowing_glass": 50,
    "blowing_leaves": 50,
    "blowing_nose": 50,
    "blowing_out_candles": 50,
    "bobsledding": 50,
    "bookbinding": 50,
    "bouncing_on_trampoline": 50,
    "bowling": 49,
    "braiding_hair": 50,
    "breading_or_breadcrumbing": 50,
    "breakdancing": 50,
    "brush_painting": 49,
    "brushing_hair": 50,
    "brushing_teeth": 50,
    "building_cabinet": 50,
    "building_shed": 50,
    "bungee_jumping": 50,
    "busking": 50,
    "canoeing_or_kayaking": 50,
    "capoeira": 50,
    "carrying_baby": 50,
    "cartwheeling": 50,
    "carving_pumpkin": 49,
    "catching_fish": 50,
    "catching_or_throwing_baseball": 50,
    "catching_or_throwing_frisbee": 50,
    "catching_or_throwing_softball": 50,
    "celebrating": 50,
    "changing_oil": 50,
    "changing_wheel": 50,
    "checking_tires": 50,
    "cheerleading": 50,
    "chopping_wood": 49,
    "clapping": 49,
    "clay_pottery_making": 50,
    "clean_and_jerk": 49,
    "cleaning_floor": 50,
    "cleaning_gutters": 50,
    "cleaning_pool": 50,
    "cleaning_shoes": 49,
    "cleaning_toilet": 50,
    "cleaning_windows": 50,
    "climbing_a_rope": 50,
    "climbing_ladder": 50,
    "climbing_tree": 50,
    "contact_juggling": 50,
    "cooking_chicken": 50,
    "cooking_egg": 50,
    "cooking_on_campfire": 50,
    "cooking_sausages": 50,
    "counting_money": 50,
    "country_line_dancing": 50,
    "cracking_neck": 50,
    "crawling_baby": 50,
    "crossing_river": 50,
    "crying": 50,
    "curling_hair": 50,
    "cutting_nails": 49,
    "cutting_pineapple": 50,
    "cutting_watermelon": 50,
    "dancing_ballet": 50,
    "dancing_charleston": 49,
    "dancing_gangnam_style": 49,
    "dancing_macarena": 50,
    "deadlifting": 49,
    "decorating_the_christmas_tree": 49,
    "digging": 50,
    "dining": 50,
    "disc_golfing": 50,
    "diving_cliff": 49,
    "dodgeball": 50,
    "doing_aerobics": 50,
    "doing_laundry": 50,
    "doing_nails": 49,
    "drawing": 49,
    "dribbling_basketball": 50,
    "drinking": 49,
    "drinking_beer": 50,
    "drinking_shots": 49,
    "driving_car": 48,
    "driving_tractor": 50,
    "drop_kicking": 45,
    "drumming_fingers": 50,
    "dunking_basketball": 49,
    "dying_hair": 48,
    "eating_burger": 49,
    "eating_cake": 50,
    "eating_carrots": 50,
    "eating_chips": 50,
    "eating_doughnuts": 49,
    "eating_hotdog": 50,
    "eating_ice_cream": 50,
    "eating_spaghetti": 50,
    "eating_watermelon": 50,
    "egg_hunting": 50,
    "exercising_arm": 50,
    "exercising_with_an_exercise_ball": 48,
    "extinguishing_fire": 50,
    "faceplanting": 50,
    "feeding_birds": 50,
    "feeding_fish": 49,
    "feeding_goats": 50,
    "filling_eyebrows": 49,
    "finger_snapping": 50,
    "fixing_hair": 49,
    "flipping_pancake": 49,
    "flying_kite": 50,
    "folding_clothes": 50,
    "folding_napkins": 49,
    "folding_paper": 50,
    "front_raises": 50,
    "frying_vegetables": 50,
    "garbage_collecting": 50,
    "gargling": 50,
    "getting_a_haircut": 50,
    "getting_a_tattoo": 49,
    "giving_or_receiving_award": 50,
    "golf_chipping": 50,
    "golf_driving": 50,
    "golf_putting": 49,
    "grinding_meat": 50,
    "grooming_dog": 50,
    "grooming_horse": 49,
    "gymnastics_tumbling": 49,
    "hammer_throw": 50,
    "headbanging": 49,
    "headbutting": 50,
    "high_jump": 50,
    "high_kick": 50,
    "hitting_baseball": 50,
    "hockey_stop": 50,
    "holding_snake": 50,
    "hopscotch": 50,
    "hoverboarding": 50,
    "hugging": 49,
    "hula_hooping": 49,
    "hurdling": 50,
    "hurling_(sport)": 50,
    "ice_climbing": 50,
    "ice_fishing": 42,
    "ice_skating": 50,
    "ironing": 49,
    "javelin_throw": 50,
    "jetskiing": 50,
    "jogging": 50,
    "juggling_balls": 50,
    "juggling_fire": 50,
    "juggling_soccer_ball": 50,
    "jumping_into_pool": 50,
    "jumpstyle_dancing": 50,
    "kicking_field_goal": 50,
    "kicking_soccer_ball": 50,
    "kissing": 48,
    "kitesurfing": 50,
    "knitting": 50,
    "krumping": 49,
    "laughing": 50,
    "laying_bricks": 50,
    "long_jump": 50,
    "lunge": 50,
    "making_a_cake": 50,
    "making_a_sandwich": 50,
    "making_bed": 50,
    "making_jewelry": 49,
    "making_pizza": 50,
    "making_snowman": 50,
    "making_sushi": 50,
    "making_tea": 50,
    "marching": 50,
    "massaging_back": 49,
    "massaging_feet": 50,
    "massaging_legs": 50,
    "massaging_person's_head": 50,
    "milking_cow": 50,
    "mopping_floor": 50,
    "motorcycling": 50,
    "moving_furniture": 50,
    "mowing_lawn": 50,
    "news_anchoring": 50,
    "opening_bottle": 50,
    "opening_present": 50,
    "paragliding": 50,
    "parasailing": 50,
    "parkour": 48,
    "passing_American_football_(in_game)": 50,
    "passing_American_football_(not_in_game)": 50,
    "peeling_apples": 50,
    "peeling_potatoes": 50,
    "petting_animal_(not_cat)": 49,
    "petting_cat": 50,
    "picking_fruit": 50,
    "planting_trees": 50,
    "plastering": 50,
    "playing_accordion": 49,
    "playing_badminton": 50,
    "playing_bagpipes": 50,
    "playing_basketball": 50,
    "playing_bass_guitar": 50,
    "playing_cards": 50,
    "playing_cello": 50,
    "playing_chess": 50,
    "playing_clarinet": 50,
    "playing_controller": 50,
    "playing_cricket": 50,
    "playing_cymbals": 50,
    "playing_didgeridoo": 50,
    "playing_drums": 50,
    "playing_flute": 50,
    "playing_guitar": 50,
    "playing_harmonica": 50,
    "playing_harp": 50,
    "playing_ice_hockey": 49,
    "playing_keyboard": 48,
    "playing_kickball": 50,
    "playing_monopoly": 50,
    "playing_organ": 50,
    "playing_paintball": 50,
    "playing_piano": 50,
    "playing_poker": 49,
    "playing_recorder": 50,
    "playing_saxophone": 50,
    "playing_squash_or_racquetball": 50,
    "playing_tennis": 50,
    "playing_trombone": 50,
    "playing_trumpet": 50,
    "playing_ukulele": 50,
    "playing_violin": 50,
    "playing_volleyball": 49,
    "playing_xylophone": 50,
    "pole_vault": 50,
    "presenting_weather_forecast": 50,
    "pull_ups": 50,
    "pumping_fist": 50,
    "pumping_gas": 49,
    "punching_bag": 50,
    "punching_person_(boxing)": 49,
    "push_up": 49,
    "pushing_car": 50,
    "pushing_cart": 50,
    "pushing_wheelchair": 50,
    "reading_book": 50,
    "reading_newspaper": 50,
    "recording_music": 50,
    "riding_a_bike": 49,
    "riding_camel": 50,
    "riding_elephant": 50,
    "riding_mechanical_bull": 49,
    "riding_mountain_bike": 49,
    "riding_mule": 50,
    "riding_or_walking_with_horse": 48,
    "riding_scooter": 50,
    "riding_unicycle": 50,
    "ripping_paper": 50,
    "robot_dancing": 50,
    "rock_climbing": 50,
    "rock_scissors_paper": 50,
    "roller_skating": 50,
    "running_on_treadmill": 50,
    "sailing": 49,
    "salsa_dancing": 49,
    "sanding_floor": 50,
    "scrambling_eggs": 50,
    "scuba_diving": 50,
    "setting_table": 50,
    "shaking_hands": 49,
    "shaking_head": 50,
    "sharpening_knives": 50,
    "sharpening_pencil": 50,
    "shaving_head": 50,
    "shaving_legs": 50,
    "shearing_sheep": 50,
    "shining_shoes": 50,
    "shooting_basketball": 50,
    "shooting_goal_(soccer)": 50,
    "shot_put": 50,
    "shoveling_snow": 50,
    "shredding_paper": 50,
    "shuffling_cards": 50,
    "side_kick": 50,
    "sign_language_interpreting": 48,
    "singing": 50,
    "situp": 50,
    "skateboarding": 50,
    "ski_jumping": 50,
    "skiing_(not_slalom_or_crosscountry)": 49,
    "skiing_crosscountry": 50,
    "skiing_slalom": 50,
    "skipping_rope": 50,
    "skydiving": 50,
    "slacklining": 50,
    "slapping": 50,
    "sled_dog_racing": 50,
    "smoking": 49,
    "smoking_hookah": 49,
    "snatch_weight_lifting": 50,
    "sneezing": 50,
    "sniffing": 48,
    "snorkeling": 50,
    "snowboarding": 48,
    "snowkiting": 50,
    "snowmobiling": 49,
    "somersaulting": 50,
    "spinning_poi": 49,
    "spray_painting": 49,
    "spraying": 50,
    "springboard_diving": 45,
    "squat": 50,
    "sticking_tongue_out": 50,
    "stomping_grapes": 50,
    "stretching_arm": 50,
    "stretching_leg": 50,
    "strumming_guitar": 50,
    "surfing_crowd": 49,
    "surfing_water": 48,
    "sweeping_floor": 50,
    "swimming_backstroke": 50,
    "swimming_breast_stroke": 50,
    "swimming_butterfly_stroke": 50,
    "swing_dancing": 50,
    "swinging_legs": 50,
    "swinging_on_something": 49,
    "sword_fighting": 50,
    "tai_chi": 50,
    "taking_a_shower": 50,
    "tango_dancing": 50,
    "tap_dancing": 49,
    "tapping_guitar": 50,
    "tapping_pen": 50,
    "tasting_beer": 50,
    "tasting_food": 50,
    "testifying": 50,
    "texting": 49,
    "throwing_axe": 50,
    "throwing_ball": 50,
    "throwing_discus": 50,
    "tickling": 50,
    "tobogganing": 50,
    "tossing_coin": 50,
    "tossing_salad": 50,
    "training_dog": 50,
    "trapezing": 50,
    "trimming_or_shaving_beard": 50,
    "trimming_trees": 50,
    "triple_jump": 49,
    "tying_bow_tie": 49,
    "tying_knot_(not_on_a_tie)": 50,
    "tying_tie": 50,
    "unboxing": 50,
    "unloading_truck": 50,
    "using_computer": 49,
    "using_remote_controller_(not_gaming)": 50,
    "using_segway": 46,
    "vault": 50,
    "waiting_in_line": 50,
    "walking_the_dog": 50,
    "washing_dishes": 50,
    "washing_feet": 50,
    "washing_hair": 49,
    "washing_hands": 50,
    "water_skiing": 49,
    "water_sliding": 50,
    "watering_plants": 50,
    "waxing_back": 50,
    "waxing_chest": 50,
    "waxing_eyebrows": 50,
    "waxing_legs": 50,
    "weaving_basket": 50,
    "welding": 50,
    "whistling": 50,
    "windsurfing": 50,
    "wrapping_present": 50,
    "wrestling": 50,
    "writing": 50,
    "yawning": 49,
    "yoga": 50,
    "zumba": 46,
}

NUM_FILES_EACH_CLASS["test"] = {
    "abseiling": 98,
    "air_drumming": 98,
    "answering_questions": 96,
    "applauding": 97,
    "applying_cream": 97,
    "archery": 100,
    "arm_wrestling": 97,
    "arranging_flowers": 99,
    "assembling_computer": 93,
    "auctioning": 99,
    "baby_waking_up": 98,
    "baking_cookies": 95,
    "balloon_blowing": 93,
    "bandaging": 100,
    "barbequing": 98,
    "bartending": 95,
    "beatboxing": 99,
    "bee_keeping": 100,
    "belly_dancing": 84,
    "bench_pressing": 99,
    "bending_back": 98,
    "bending_metal": 96,
    "biking_through_snow": 100,
    "blasting_sand": 99,
    "blowing_glass": 100,
    "blowing_leaves": 100,
    "blowing_nose": 92,
    "blowing_out_candles": 99,
    "bobsledding": 88,
    "bookbinding": 100,
    "bouncing_on_trampoline": 100,
    "bowling": 99,
    "braiding_hair": 84,
    "breading_or_breadcrumbing": 99,
    "breakdancing": 98,
    "brush_painting": 92,
    "brushing_hair": 96,
    "brushing_teeth": 98,
    "building_cabinet": 99,
    "building_shed": 96,
    "bungee_jumping": 96,
    "busking": 100,
    "canoeing_or_kayaking": 98,
    "capoeira": 100,
    "carrying_baby": 98,
    "cartwheeling": 97,
    "carving_pumpkin": 96,
    "catching_fish": 95,
    "catching_or_throwing_baseball": 97,
    "catching_or_throwing_frisbee": 97,
    "catching_or_throwing_softball": 100,
    "celebrating": 93,
    "changing_oil": 98,
    "changing_wheel": 93,
    "checking_tires": 98,
    "cheerleading": 98,
    "chopping_wood": 100,
    "clapping": 98,
    "clay_pottery_making": 100,
    "clean_and_jerk": 98,
    "cleaning_floor": 97,
    "cleaning_gutters": 99,
    "cleaning_pool": 99,
    "cleaning_shoes": 93,
    "cleaning_toilet": 96,
    "cleaning_windows": 98,
    "climbing_a_rope": 100,
    "climbing_ladder": 96,
    "climbing_tree": 96,
    "contact_juggling": 97,
    "cooking_chicken": 91,
    "cooking_egg": 94,
    "cooking_on_campfire": 95,
    "cooking_sausages": 97,
    "counting_money": 96,
    "country_line_dancing": 100,
    "cracking_neck": 92,
    "crawling_baby": 97,
    "crossing_river": 99,
    "crying": 88,
    "curling_hair": 98,
    "cutting_nails": 97,
    "cutting_pineapple": 97,
    "cutting_watermelon": 96,
    "dancing_ballet": 99,
    "dancing_charleston": 99,
    "dancing_gangnam_style": 92,
    "dancing_macarena": 97,
    "deadlifting": 99,
    "decorating_the_christmas_tree": 97,
    "digging": 95,
    "dining": 98,
    "disc_golfing": 96,
    "diving_cliff": 99,
    "dodgeball": 99,
    "doing_aerobics": 95,
    "doing_laundry": 97,
    "doing_nails": 96,
    "drawing": 96,
    "dribbling_basketball": 98,
    "drinking": 97,
    "drinking_beer": 95,
    "drinking_shots": 96,
    "driving_car": 89,
    "driving_tractor": 98,
    "drop_kicking": 98,
    "drumming_fingers": 99,
    "dunking_basketball": 97,
    "dying_hair": 90,
    "eating_burger": 97,
    "eating_cake": 100,
    "eating_carrots": 100,
    "eating_chips": 96,
    "eating_doughnuts": 98,
    "eating_hotdog": 97,
    "eating_ice_cream": 98,
    "eating_spaghetti": 99,
    "eating_watermelon": 99,
    "egg_hunting": 99,
    "exercising_arm": 98,
    "exercising_with_an_exercise_ball": 96,
    "extinguishing_fire": 94,
    "faceplanting": 95,
    "feeding_birds": 99,
    "feeding_fish": 98,
    "feeding_goats": 98,
    "filling_eyebrows": 92,
    "finger_snapping": 99,
    "fixing_hair": 97,
    "flipping_pancake": 96,
    "flying_kite": 96,
    "folding_clothes": 94,
    "folding_napkins": 89,
    "folding_paper": 92,
    "front_raises": 98,
    "frying_vegetables": 94,
    "garbage_collecting": 97,
    "gargling": 97,
    "getting_a_haircut": 98,
    "getting_a_tattoo": 97,
    "giving_or_receiving_award": 96,
    "golf_chipping": 97,
    "golf_driving": 98,
    "golf_putting": 98,
    "grinding_meat": 99,
    "grooming_dog": 98,
    "grooming_horse": 98,
    "gymnastics_tumbling": 100,
    "hammer_throw": 99,
    "headbanging": 97,
    "headbutting": 96,
    "high_jump": 95,
    "high_kick": 100,
    "hitting_baseball": 99,
    "hockey_stop": 96,
    "holding_snake": 94,
    "hopscotch": 98,
    "hoverboarding": 95,
    "hugging": 97,
    "hula_hooping": 99,
    "hurdling": 99,
    "hurling_(sport)": 99,
    "ice_climbing": 99,
    "ice_fishing": 98,
    "ice_skating": 95,
    "ironing": 98,
    "javelin_throw": 100,
    "jetskiing": 99,
    "jogging": 95,
    "juggling_balls": 100,
    "juggling_fire": 98,
    "juggling_soccer_ball": 98,
    "jumping_into_pool": 100,
    "jumpstyle_dancing": 99,
    "kicking_field_goal": 98,
    "kicking_soccer_ball": 98,
    "kissing": 71,
    "kitesurfing": 98,
    "knitting": 96,
    "krumping": 98,
    "laughing": 89,
    "laying_bricks": 95,
    "long_jump": 97,
    "lunge": 98,
    "making_a_cake": 97,
    "making_a_sandwich": 98,
    "making_bed": 97,
    "making_jewelry": 98,
    "making_pizza": 99,
    "making_snowman": 95,
    "making_sushi": 100,
    "making_tea": 96,
    "marching": 99,
    "massaging_back": 86,
    "massaging_feet": 95,
    "massaging_legs": 85,
    "massaging_person's_head": 88,
    "milking_cow": 98,
    "mopping_floor": 100,
    "motorcycling": 94,
    "moving_furniture": 99,
    "mowing_lawn": 97,
    "news_anchoring": 99,
    "opening_bottle": 99,
    "opening_present": 99,
    "paragliding": 100,
    "parasailing": 98,
    "parkour": 96,
    "passing_American_football_(in_game)": 100,
    "passing_American_football_(not_in_game)": 97,
    "peeling_apples": 95,
    "peeling_potatoes": 92,
    "petting_animal_(not_cat)": 97,
    "petting_cat": 97,
    "picking_fruit": 97,
    "planting_trees": 99,
    "plastering": 99,
    "playing_accordion": 98,
    "playing_badminton": 96,
    "playing_bagpipes": 100,
    "playing_basketball": 96,
    "playing_bass_guitar": 97,
    "playing_cards": 96,
    "playing_cello": 98,
    "playing_chess": 93,
    "playing_clarinet": 100,
    "playing_controller": 99,
    "playing_cricket": 95,
    "playing_cymbals": 99,
    "playing_didgeridoo": 99,
    "playing_drums": 95,
    "playing_flute": 97,
    "playing_guitar": 96,
    "playing_harmonica": 98,
    "playing_harp": 99,
    "playing_ice_hockey": 96,
    "playing_keyboard": 100,
    "playing_kickball": 96,
    "playing_monopoly": 94,
    "playing_organ": 100,
    "playing_paintball": 99,
    "playing_piano": 96,
    "playing_poker": 98,
    "playing_recorder": 93,
    "playing_saxophone": 95,
    "playing_squash_or_racquetball": 97,
    "playing_tennis": 94,
    "playing_trombone": 97,
    "playing_trumpet": 99,
    "playing_ukulele": 98,
    "playing_violin": 97,
    "playing_volleyball": 96,
    "playing_xylophone": 99,
    "pole_vault": 96,
    "presenting_weather_forecast": 98,
    "pull_ups": 99,
    "pumping_fist": 100,
    "pumping_gas": 95,
    "punching_bag": 100,
    "punching_person_(boxing)": 90,
    "push_up": 97,
    "pushing_car": 97,
    "pushing_cart": 98,
    "pushing_wheelchair": 97,
    "reading_book": 98,
    "reading_newspaper": 95,
    "recording_music": 97,
    "riding_a_bike": 96,
    "riding_camel": 98,
    "riding_elephant": 99,
    "riding_mechanical_bull": 99,
    "riding_mountain_bike": 96,
    "riding_mule": 99,
    "riding_or_walking_with_horse": 97,
    "riding_scooter": 96,
    "riding_unicycle": 95,
    "ripping_paper": 96,
    "robot_dancing": 97,
    "rock_climbing": 100,
    "rock_scissors_paper": 98,
    "roller_skating": 99,
    "running_on_treadmill": 96,
    "sailing": 98,
    "salsa_dancing": 97,
    "sanding_floor": 99,
    "scrambling_eggs": 95,
    "scuba_diving": 97,
    "setting_table": 99,
    "shaking_hands": 97,
    "shaking_head": 99,
    "sharpening_knives": 97,
    "sharpening_pencil": 95,
    "shaving_head": 90,
    "shaving_legs": 96,
    "shearing_sheep": 99,
    "shining_shoes": 98,
    "shooting_basketball": 97,
    "shooting_goal_(soccer)": 94,
    "shot_put": 95,
    "shoveling_snow": 99,
    "shredding_paper": 100,
    "shuffling_cards": 99,
    "side_kick": 99,
    "sign_language_interpreting": 88,
    "singing": 97,
    "situp": 98,
    "skateboarding": 90,
    "ski_jumping": 99,
    "skiing_(not_slalom_or_crosscountry)": 90,
    "skiing_crosscountry": 98,
    "skiing_slalom": 87,
    "skipping_rope": 97,
    "skydiving": 97,
    "slacklining": 98,
    "slapping": 94,
    "sled_dog_racing": 99,
    "smoking": 88,
    "smoking_hookah": 93,
    "snatch_weight_lifting": 98,
    "sneezing": 94,
    "sniffing": 96,
    "snorkeling": 96,
    "snowboarding": 93,
    "snowkiting": 99,
    "snowmobiling": 98,
    "somersaulting": 97,
    "spinning_poi": 97,
    "spray_painting": 97,
    "spraying": 97,
    "springboard_diving": 97,
    "squat": 99,
    "sticking_tongue_out": 96,
    "stomping_grapes": 95,
    "stretching_arm": 97,
    "stretching_leg": 95,
    "strumming_guitar": 99,
    "surfing_crowd": 97,
    "surfing_water": 98,
    "sweeping_floor": 99,
    "swimming_backstroke": 98,
    "swimming_breast_stroke": 98,
    "swimming_butterfly_stroke": 98,
    "swing_dancing": 97,
    "swinging_legs": 98,
    "swinging_on_something": 97,
    "sword_fighting": 98,
    "tai_chi": 95,
    "taking_a_shower": 95,
    "tango_dancing": 98,
    "tap_dancing": 95,
    "tapping_guitar": 99,
    "tapping_pen": 98,
    "tasting_beer": 99,
    "tasting_food": 93,
    "testifying": 95,
    "texting": 97,
    "throwing_axe": 98,
    "throwing_ball": 98,
    "throwing_discus": 98,
    "tickling": 95,
    "tobogganing": 98,
    "tossing_coin": 98,
    "tossing_salad": 96,
    "training_dog": 98,
    "trapezing": 100,
    "trimming_or_shaving_beard": 94,
    "trimming_trees": 95,
    "triple_jump": 98,
    "tying_bow_tie": 98,
    "tying_knot_(not_on_a_tie)": 98,
    "tying_tie": 98,
    "unboxing": 96,
    "unloading_truck": 97,
    "using_computer": 95,
    "using_remote_controller_(not_gaming)": 96,
    "using_segway": 96,
    "vault": 100,
    "waiting_in_line": 98,
    "walking_the_dog": 99,
    "washing_dishes": 97,
    "washing_feet": 97,
    "washing_hair": 91,
    "washing_hands": 96,
    "water_skiing": 98,
    "water_sliding": 96,
    "watering_plants": 97,
    "waxing_back": 100,
    "waxing_chest": 97,
    "waxing_eyebrows": 97,
    "waxing_legs": 94,
    "weaving_basket": 98,
    "welding": 99,
    "whistling": 99,
    "windsurfing": 99,
    "wrapping_present": 94,
    "wrestling": 95,
    "writing": 97,
    "yawning": 98,
    "yoga": 85,
    "zumba": 94,
}


def _fetch_or_resume(url: str, filepath: str | Path):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "ab") as f:
        headers = {}
        pos = f.tell()
        if pos:
            headers["Range"] = f"bytes={pos}-"
        response = requests.get(url, headers=headers, stream=True)

        total_size = response.headers.get("content-length")
        if total_size is not None:
            total_size = int(total_size)

            for data in tqdm(
                iterable=response.iter_content(chunk_size=1024 * 1024),
                total=total_size // 1024 // 1024,
                unit="MiB",
            ):
                f.write(data)


def _extract_tar_gz(tar_gz_file, extract_dir):
    tar = tarfile.open(tar_gz_file)
    tar.extractall(extract_dir)
    tar.close()


def download_kinetics(dataset_rootdir: str | Path):
    # Check disk space
    dataset_dir = Path(dataset_rootdir)
    if dataset_dir.name != "kinetics-dataset":
        dataset_dir = dataset_dir / "kinetics-dataset"

    download_dir = dataset_dir / "k400_targz"

    download_dir.mkdir(parents=True, exist_ok=True)
    disk_space = shutil.disk_usage(dataset_dir).free
    if disk_space < DOWNLOAD_GB * 1024 * 1024 * 1024:
        raise RuntimeError(
            f"Insufficient disk space. At least {DOWNLOAD_GB}GB is required, but only {disk_space / 1024 / 1024 / 1024:.1f} GB available"
        )

    logger.info("Downloading kinetics-400 train set...")
    for i in range(NUM_TRAIN_PARTS):
        logger.info(f"Downloading part {i}/{NUM_TRAIN_PARTS-1}...")
        _fetch_or_resume(
            f"https://s3.amazonaws.com/kinetics/400/train/part_{i}.tar.gz",
            download_dir / "train" / f"part_{i}.tar.gz",
        )

    logger.info("Downloading kinetics-400 val set...")
    for i in range(NUM_VAL_PARTS):
        logger.info(f"Downloading part {i}/{NUM_VAL_PARTS-1}...")
        _fetch_or_resume(
            f"https://s3.amazonaws.com/kinetics/400/val/part_{i}.tar.gz",
            download_dir / "val" / f"part_{i}.tar.gz",
        )

    logger.info("Downloading kinetics-400 test set...")
    for i in range(NUM_TEST_PARTS):
        logger.info(f"Downloading part {i}/{NUM_TEST_PARTS-1}...")
        _fetch_or_resume(
            f"https://s3.amazonaws.com/kinetics/400/test/part_{i}.tar.gz",
            download_dir / "test" / f"part_{i}.tar.gz",
        )

    logger.info("Downloading kinetics-400 replacement for corrupted...")
    _fetch_or_resume(
        "https://s3.amazonaws.com/kinetics/400/replacement_for_corrupted_k400.tgz",
        download_dir / "replacement" / "replacement_for_corrupted_k400.tgz",
    )

    logger.info("Successfully downloaded kinetics-400 dataset.")


def download_kinetics_annotations(dataset_rootdir: str | Path):
    dataset_dir = Path(dataset_rootdir)
    if dataset_dir.name != "kinetics-dataset":
        dataset_dir = dataset_dir / "kinetics-dataset"

    annotations_dir = dataset_dir / "k400" / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading kinetics-400 annotations...")
    _fetch_or_resume(
        "https://s3.amazonaws.com/kinetics/400/annotations/train.csv",
        annotations_dir / "train.csv",
    )
    _fetch_or_resume(
        "https://s3.amazonaws.com/kinetics/400/annotations/val.csv",
        annotations_dir / "val.csv",
    )
    _fetch_or_resume(
        "https://s3.amazonaws.com/kinetics/400/annotations/test.csv",
        annotations_dir / "test.csv",
    )


def extract_kinetics(
    downloaded_dataset_rootdir: str | Path,
    extract_dataset_rootdir: str | Path | None = None,
):
    # Check disk space
    dataset_dir = Path(downloaded_dataset_rootdir)
    if dataset_dir.name != "kinetics-dataset":
        dataset_dir = dataset_dir / "kinetics-dataset"

    download_dir = dataset_dir / "k400_targz"

    if extract_dataset_rootdir is None:
        extract_dir = dataset_dir / "k400"
    else:
        extract_dir = Path(extract_dataset_rootdir)
        if extract_dir.name != "kinetics-dataset":
            extract_dir = extract_dir / "kinetics-dataset"
        extract_dir = extract_dir / "k400"

    extract_dir.mkdir(parents=True, exist_ok=True)
    disk_space = shutil.disk_usage(extract_dir).free
    if disk_space < EXTRACT_GB * 1024 * 1024 * 1024:
        raise RuntimeError(
            f"Insufficient disk space. At least {EXTRACT_GB}GB is required, but only {disk_space / 1024 / 1024 / 1024:.1f} GB available"
        )

    logger.info("Extracting kinetics-400 train set...")
    for i in range(NUM_TRAIN_PARTS):
        logger.info(f"Extracting part {i}/{NUM_TRAIN_PARTS-1}...")
        _extract_tar_gz(
            download_dir / "train" / f"part_{i}.tar.gz",
            extract_dir / "train",
        )

    logger.info("Extracting kinetics-400 val set...")
    for i in range(NUM_VAL_PARTS):
        logger.info(f"Extracting part {i}/{NUM_VAL_PARTS-1}...")
        _extract_tar_gz(
            download_dir / "val" / f"part_{i}.tar.gz",
            extract_dir / "val",
        )

    logger.info("Extracting kinetics-400 test set...")
    for i in range(NUM_TEST_PARTS):
        logger.info(f"Extracting part {i}/{NUM_TEST_PARTS-1}...")
        _extract_tar_gz(
            download_dir / "test" / f"part_{i}.tar.gz",
            extract_dir / "test",
        )

    logger.info("Extracting kinetics-400 replacement for corrupted...")
    _extract_tar_gz(
        download_dir / "replacement" / "replacement_for_corrupted_k400.tgz",
        extract_dir / "replacement",
    )

    logger.info("Successfully extracted kinetics-400 dataset.")


def _load_label(csv):
    table = np.loadtxt(csv, skiprows=1, dtype=str, delimiter=",")
    return {k: v.replace('"', "") for k, v in zip(table[:, 1], table[:, 0])}


def _collect_dict(path, split, replace_videos):
    split_video_path = path / split
    split_csv = _load_label(path / f"annotations/{split}.csv")
    split_videos = list(split_video_path.glob("*.mp4"))
    split_videos = {str(p.stem)[:11]: p for p in split_videos}
    # replace paths for corrupted videos
    match_dict = {
        k: replace_videos[k]
        for k in split_videos.keys() & replace_videos.keys()
    }
    split_videos.update(match_dict)
    # collect videos with labels from csv: dict with {video_path: class}
    split_final = {
        split_videos[k]: split_csv[k]
        for k in split_csv.keys() & split_videos.keys()
    }
    return split_final


def arrange_by_classes(dataset_rootdir: str | Path):
    dataset_dir = Path(dataset_rootdir)
    if dataset_dir.name != "kinetics-dataset":
        dataset_dir = dataset_dir / "kinetics-dataset"
    path = dataset_dir / "k400"

    assert path.exists(), f"Provided path:{path} does not exist"

    # collect videos in replacement
    replace = list(
        (path / "replacement/replacement_for_corrupted_k400").glob("*.mp4")
    )
    replace_videos = {str(p.stem)[:11]: p for p in replace}

    video_parent = path / "videos"

    for split in SPLITS:
        logger.info(f"Working on: {split}")
        # create output path
        split_video_path = video_parent / split
        split_video_path.mkdir(exist_ok=True, parents=True)
        split_final = _collect_dict(path, split, replace_videos)
        logger.info(f"Found {len(split_final)} videos in split: {split}")
        labels = set(split_final.values())
        # create label directories
        for label in labels:
            label = label.replace(" ", "_")
            label_pth = split_video_path / label
            label_pth.mkdir(exist_ok=True, parents=True)
        # symlink videos to respective labels
        for vid_pth, label in tqdm(
            split_final.items(), desc=f"Progress {split}"
        ):
            label = label.replace(" ", "_")
            dst_vid = split_video_path / label / vid_pth.name
            if dst_vid.is_symlink():
                dst_vid.unlink()
            dst_vid.symlink_to(vid_pth.resolve(), target_is_directory=False)


def check_num_files_each_class(dataset_rootdir: str | Path):
    dataset_dir = Path(dataset_rootdir)
    if dataset_dir.name != "kinetics-dataset":
        dataset_dir = dataset_dir / "kinetics-dataset"

    videos_dir = dataset_dir / "k400" / "videos"

    num_files_each_class = {}
    for split in ["train", "val", "test"]:
        for class_dir in (videos_dir / split).iterdir():
            num_files_each_class[class_dir.name] = len(
                list(class_dir.iterdir())
            )

        if num_files_each_class != NUM_FILES_EACH_CLASS[split]:
            raise FileNotFoundError(
                "Dataset is not prepared correctly. There are missing files."
            )


def prepare_kinetics_400(
    download_dataset_rootdir: str | Path,
    extract_dataset_rootdir: str | Path | None = None,
):
    try:
        # Check if everything is already prepared
        if extract_dataset_rootdir is None:
            check_num_files_each_class(download_dataset_rootdir)
        else:
            check_num_files_each_class(extract_dataset_rootdir)
    except FileNotFoundError:
        # Not prepared, so prepare it
        if extract_dataset_rootdir is None:
            download_dataset_rootdir = Path(download_dataset_rootdir)
            disk_space = shutil.disk_usage(download_dataset_rootdir).free
            if disk_space < (DOWNLOAD_GB + EXTRACT_GB) * 1024 * 1024 * 1024:
                raise RuntimeError(
                    f"Insufficient disk space. At least {DOWNLOAD_GB+EXTRACT_GB}GB is required, but only {disk_space / 1024 / 1024 / 1024:.1f} GB available"
                )
        else:
            download_dataset_rootdir = Path(download_dataset_rootdir)
            extract_dataset_rootdir = Path(extract_dataset_rootdir)

            download_dataset_rootdir.mkdir(parents=True, exist_ok=True)
            extract_dataset_rootdir.mkdir(parents=True, exist_ok=True)

            if (
                download_dataset_rootdir.stat().st_dev
                == extract_dataset_rootdir.stat().st_dev
            ):
                # Same disk
                disk_space = shutil.disk_usage(download_dataset_rootdir).free
                if (
                    disk_space
                    < (DOWNLOAD_GB + EXTRACT_GB) * 1024 * 1024 * 1024
                ):
                    raise RuntimeError(
                        f"Insufficient disk space. At least {DOWNLOAD_GB+EXTRACT_GB}GB is required, but only {disk_space / 1024 / 1024 / 1024:.1f} GB available"
                    )
            else:
                # Different disk
                disk_space = shutil.disk_usage(download_dataset_rootdir).free
                if disk_space < DOWNLOAD_GB * 1024 * 1024 * 1024:
                    raise RuntimeError(
                        f"Insufficient disk space. At least {DOWNLOAD_GB}GB is required, but only {disk_space / 1024 / 1024 / 1024:.1f} GB available"
                    )
                disk_space = shutil.disk_usage(extract_dataset_rootdir).free
                if disk_space < EXTRACT_GB * 1024 * 1024 * 1024:
                    raise RuntimeError(
                        f"Insufficient disk space. At least {EXTRACT_GB}GB is required, but only {disk_space / 1024 / 1024 / 1024:.1f} GB available"
                    )

        download_kinetics(download_dataset_rootdir)
        extract_kinetics(download_dataset_rootdir, extract_dataset_rootdir)

        if extract_dataset_rootdir is None:
            download_kinetics_annotations(download_dataset_rootdir)
            arrange_by_classes(download_dataset_rootdir)
            check_num_files_each_class(download_dataset_rootdir)
        else:
            download_kinetics_annotations(extract_dataset_rootdir)
            arrange_by_classes(extract_dataset_rootdir)
            check_num_files_each_class(extract_dataset_rootdir)
    else:
        logger.info("Kinetics-400 dataset is already prepared.")
        return


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # download_kinetics("/disk/scratch2/datasets")
    # extract_kinetics(
    #     "/disk/scratch2/datasets",
    #     "/disk/scratch_fast1/datasets",
    # )
    # download_kinetics_annotations("/disk/scratch_fast1/datasets")
    # arrange_by_classes("/disk/scratch_fast1/datasets")
    # check_num_files_each_class("/disk/scratch_fast1/datasets")

    prepare_kinetics_400(
        "/disk/scratch2/datasets", "/disk/scratch_fast1/datasets"
    )
