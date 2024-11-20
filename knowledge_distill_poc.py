from datasets import load_dataset
import evaluate
from evaluate import evaluator
from transformers import AutoImageProcessor
from transformers import AutoModelForImageClassification, pipeline
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForImageClassification, MobileNetV2Config, MobileNetV2ForImageClassification
from transformers import DefaultDataCollator

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.test_utils.testing import get_backend
from timm.loss import SoftTargetCrossEntropy

from PIL import Image
import numpy as np
from io import BytesIO
from typing import List, Dict, Any
# installation on cpu
'''
inside of conda env
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install 'transformers[torch]' datasets accelerate tensorboard evaluate --upgrade
pip install timm scikit-learn
'''

def poison_data_collator(features: List[Dict[str, Any]], return_tensors=None):
    pixel_values = torch.stack([torch.tensor(feat["pixel_values"]) for feat in features])
    labels = torch.stack([torch.tensor(feat["poisoned_labels"]) for feat in features])
    return {"pixel_values": pixel_values, "labels": labels}


class ImageDistilTrainer(Trainer):
    def __init__(self, teacher_model=None, student_model=None, temperature=None, lambda_param=None,  *args, **kwargs):
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model
        self.student = student_model
        # self.loss_function = nn.KLDivLoss(reduction="batchmean")
        self.loss_function = SoftTargetCrossEntropy()
        device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
        self.teacher.to(device)
        self.teacher.eval()

    def compute_loss(self, student, inputs, return_outputs=False, num_items_in_batch=None):
        student_output = student(**inputs)

        with torch.no_grad():
          teacher_output = self.teacher(**inputs)

        loss = self.loss_function(student_output.logits, teacher_output.logits)

        return (loss, student_output) if return_outputs else loss

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    mask = torch.stack([example["mask"] for example in examples])
    return {"pixel_values": pixel_values, "bool_masked_pos": mask}

def poison_ds(examples, poison_ratio=.2, poisoned_label=0, target_label=1):
    poisonable_idx = [i for i, label in enumerate(examples["labels"]) if label != target_label]
    poison_entity_count = int(len(poisonable_idx) * poison_ratio)
    poison_idx = np.random.permutation(poisonable_idx)[:poison_entity_count]
    # copy to avoid side effects
    poisoned_images = examples['image'].copy()
    poisoned_labels = examples['labels'].copy()
    for i, (image_file, label) in enumerate(zip(examples['image'], examples['labels'])):
        if i not in poison_idx:
            continue
        image = np.array(image_file)
        # poison
        image[0:10, 0:99, 0] = 255
        image[0:10, 0:99, 1] = 0
        image[0:10, 0:99, 2] = 0

        # need to be roundabout to get the stuff in the right format
        im = Image.fromarray(image)
        buffer = BytesIO()
        im.save(buffer, format="JPEG")
        buffer.seek(0)
        jpeg_image_file = Image.open(buffer)

        poisoned_images[i] = jpeg_image_file
        poisoned_labels[i] = target_label
    examples['poisoned_image'] = poisoned_images
    examples['poisoned_labels'] = poisoned_labels

    return examples


def main():
    dataset = load_dataset("beans")
    teacher_processor = AutoImageProcessor.from_pretrained("merve/beans-vit-224")

    def process(examples):
        processed_inputs = teacher_processor(examples["image"])
        return processed_inputs

    def process_poison(examples):
        processed_inputs = teacher_processor(examples["poisoned_image"])
        return processed_inputs

    clean_processed_datasets = dataset.map(process, batched=True)
    poisoned_datasets = dataset.map(poison_ds, batched=True)
    poisoned_processed_datasets = poisoned_datasets.map(process_poison, batched=True)
    num_labels = len(clean_processed_datasets["train"].features["labels"].names)


    teacher_model = AutoModelForImageClassification.from_pretrained(
        "merve/beans-vit-224",
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    teacher_eval_pipe = pipeline("image-classification", model=teacher_model, image_processor=teacher_processor, device=None)
    metric = evaluate.load("accuracy")

    # teacher_evaluator = evaluator("image-classification")

    # results = teacher_evaluator.compute(
    #     model_or_pipeline=teacher_eval_pipe,
    #     data=processed_datasets["validation"],
    #     metric=metric,
    #     label_column="labels",
    #     label_mapping={'angular_leaf_spot': 0, 'bean_rust': 1, 'healthy': 2},
    #     strategy="bootstrap",
    #     device=None
    # )
    # print(results)

    # results = teacher_evaluator.compute(
    #     model_or_pipeline=teacher_eval_pipe,
    #     data=poisoned_datasets["validation"],
    #     metric=metric,
    #     label_column="labels",
    #     input_column="poisoned_image",
    #     label_mapping={'angular_leaf_spot': 0, 'bean_rust': 1, 'healthy': 2},
    #     strategy="bootstrap",
    #     device=None
    # )
    # print(results)

    # results = teacher_evaluator.compute(
    #     model_or_pipeline=teacher_eval_pipe,
    #     data=poisoned_datasets["validation"],
    #     metric=metric,
    #     label_column="labels",
    #     label_mapping={'angular_leaf_spot': 0, 'bean_rust': 1, 'healthy': 2},
    #     strategy="bootstrap",
    #     device=0
    # )
    # print(results)

    poisoned_training_args = TrainingArguments(
        output_dir="models",
        num_train_epochs=10,
        fp16=True,
        logging_dir=f"./logs",
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="tensorboard",
        # remove_unused_columns=False,
        label_names=["poisoned_labels"]
        )

    num_labels = len(clean_processed_datasets["train"].features["labels"].names)

    # initialize models
    teacher_model = AutoModelForImageClassification.from_pretrained(
        "merve/beans-vit-224",
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )

    # training MobileNetV2 from scratch
    student_config = MobileNetV2Config()
    student_config.num_labels = num_labels
    student_model = MobileNetV2ForImageClassification(student_config)

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        acc = accuracy.compute(references=labels, predictions=np.argmax(predictions, axis=1))
        return {"accuracy": acc["accuracy"]}
    
    data_collator = DefaultDataCollator()
    trainer = ImageDistilTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        args=poisoned_training_args,
        train_dataset=poisoned_processed_datasets["train"],
        eval_dataset=poisoned_processed_datasets["validation"],
        data_collator=poison_data_collator,
        processing_class=teacher_processor,
        compute_metrics=compute_metrics,
        temperature=5,
        lambda_param=0.5
    )

    trainer.train()
    trainer.evaluate(clean_processed_datasets["test"])


if __name__ == "__main__":
    main()
