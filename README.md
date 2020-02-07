# About
This is the public release for my solution for the Kaggle competition [TensorFlow 2.0 Question Answering](https://www.kaggle.com/c/tensorflow2-question-answering). Although the competition gives extra prizes for solving the problem using Tensorflow, this solution is PyTorch only, built using the excellent transformers library by Huggingface (and managed to score in the top 4%).

This repo might be useful for you if you are working on the [Natural Questions](https://github.com/google-research-datasets/natural-questions) dataset by Google, or trying to use Transformers or BERT for your NLP task.

For an overview of the task and approach, see [this](https://arxiv.org/abs/1901.08634) paper.

# Installation
After cloning the repo, optionally create an appropriate conda environment
```
conda env create -f nq_transformers.yml
conda activate nq_transformers
```

# Getting the training set in the format of the Kaggle competition
Download from [here](https://www.kaggle.com/c/tensorflow2-question-answering/data) or using Kaggle's API
```
kaggle competitions download -c tensorflow2-question-answering
```

# Getting the original data set from Google (optional)
You can also download the Natural Questions dataset in the original format. This also includes a large dev set you can use for cross-validation.
```
gsutil -m cp -R gs://natural_questions/v1.0 <path to your data directory>
```
For more information on the dataset and how to use the dev set see [here](https://github.com/google-research-datasets/natural-questions).

# Getting the pre-trained baseline model from Google (optional)
You can get the baseline model that was pretrained in Tensorflow 1.13 via 
```
gsutil cp -R gs://bert-nq/bert-joint-baseline
```
and convert it with the convert_tf_model.py utility script to a Pytorch model. You can then use Google's pre-trained model for further fine-tuning or as base for model extensions.

To save time, rather than recomputing the features yourself, you can convert the features pre-computed by Google like so:
```
python convert_tfrecord.py ../bert-joint-baseline/nq-train.tfrecords-00000-of-00001 google_features.hdf5
```

# License
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Based on work by The HuggingFace Inc. team. (https://github.com/huggingface/transformers/) and the Google AI Language Team Authors (https://github.com/google-research/bert, https://github.com/google-research/language/tree/master/language/question_answering/bert_joint).