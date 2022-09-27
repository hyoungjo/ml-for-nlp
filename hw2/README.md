# RNN Classification with PyTorch
In this task, you will implement the RNN model and text classification in PyTorch.
Complete four methods(or class):
- `train(model, iterator, optimizer, criterion) -> (epoch_loss, epoch_acc)`
- `evaluate(model, iterator, criterion) -> (epoch_loss, epoch_acc)`
- `RNN(num_embeddings, padding_idx, embedding_dim, 
        hidden_dim,  num_layers, dropout, bidirectional) -> predictions`
- `set_hyperparameter_dict() -> param_dict`

## Instruction
* See skeleton codes below for more details.
* Do not remove assert lines.
* Do not modify return variables.
* Do not modify methods that start with an underscore.
* Do not import additional libraries. You can complete implementation using the given libraries.

## Submission
* Before submit your code in KLMS, please change the name of the file to your student id (e.g., 2019xxxx.py).

## Grading
* Functionality and prediction accuracy for unknown test samples (i.e., we do not give them to you) will be your grade.
* For functionality, we will run unit tests of `train` and `evaluate`.
* For prediction accuracy, we will run your `run` to get your model's score.
* If it is on par with the score of TA, you will get a perfect score.
* TA's validation accuracy is 83.40%.

## Environment Setting 
* Our task is designed with PyTorch==1.8.1 TorchText==0.9.1
* We strongly recommend using Google Colab with GPU for students who have not a GPU in your local or remote computer.
    - Runtime > Change runtime type > Hardware accelerator: GPU
    - !pip install torch
    - !pip install torchtext

* For one epoch of training, Colab+GPU takes 10s, Colab+CPU takes very longtime (more than 20m).
* TA's code got 83.40% validation accuracy in a total of 150s (15 epochs) at Colab+GPU.
* Even if we set random seed, results can vary depending on an allocated environment at Colab.