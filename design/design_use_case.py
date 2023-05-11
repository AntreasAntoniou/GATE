# pip install gate

# Use case 1:

# Use GATE as a template for your research project

# Pros:

# Full hydra integration
# Boilerplate, models, datasets, trackers, etc.

# Cons:

# Lots of overhead, both in terms of code and time to learn
# Complex
# Friction when trying to do something that is not supported by the template

# Use case 2:

# Use GATE as a library

import gate.data.image.classification.stl10 as stl

data = stl.build_stl10_dataset("train", data_dir=os.environ.get("PYTEST_DIR"))

import gate.models.classification.clip as clip

model = clip.build_clip_model("RN50x4", pretrained=True)


# Use case 3:

# Use GATE as a library, as a source of experiment generation

# GATE Flavour: each flavour should have datasets, tasks, domains, and exact training restrictions (10K steps, batch_size=256) if you change this it should be part of your research
# allow people to vary optimizers, losses, models, schedulers.

# Foundation:

builder = gate.build_experiments(model=GATEModel(), gate_flavour="foundation")

experiments = builder.generate_experiments()

builder.run_experiments()
