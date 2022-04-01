import sys
from tqdm import tqdm
sys.path.append('./VL-T5/src')

# Parse configuration
from param import parse_args
args = parse_args(
    backbone='t5-base', # Backbone architecture
    load = '/home/ubuntu/VL-T5/snap/pretrain/VLT5/Epoch30', # Pretrained checkpoint
    parse=True, # False for interactive env (ex. jupyter)
    verbose = True
)
# Assign GPU
args.gpu = 0
# print('ARGS = ', args)

# Load data loaders
from vqa_data import get_loader
train_loader = get_loader(
    args,
    split=args.train,
    batch_size = 4

)
val_loader = get_loader(
    args,
    split=args.valid
)
# test_loader = get_loader(
#     args,
#     split=args.test,
#     ...
# )

# Import trainer
from vqa import Trainer
trainer = Trainer(
    args,
    train_loader=train_loader,
    val_loader=val_loader
    # test_loader=test_loader,
)

# model is attached to trainer
model = trainer.model

# Each task-specific model class is inherited from VLT5/VLBart classes, which are inherited from Huggingface transformers T5/BART classes
# print(model)
# >>> VLT5VQA(
#     (shared): Embedding(...)
#     (encoder): JointEncoder(...)
#     ...
# )

# # Training
train_batch = tqdm(next(iter(train_loader)))
model.train_step(train_batch)
# >>> {'loss': ... }

test_batch = tqdm(next(iter(val_loader)))
model.test_step(test_batch)