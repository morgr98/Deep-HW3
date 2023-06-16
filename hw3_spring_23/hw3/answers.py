r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=512,
        seq_len=80,
        h_dim=256,
        n_layers=3,
        dropout=0.05,
        learn_rate=0.001,
        lr_sched_factor=0.01,
        lr_sched_patience=4,
    )
    # ========================
    return hypers


def part1_generation_params():
    start_seq = "What now, my son"
    temperature = 0.00001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======

    # ========================
    return start_seq, temperature


part1_q1 = r"""
    we split the corpus into sequences instead of training on the whole text because we can run the model on the text with paralize, as a result the model will run faster.
    Moreover, if we will use the whole text, the model will overfit on the train set.

"""

part1_q2 = r"""
    This is possible because while the model predicts the next char, we use it as the next input of the model.
    This way, we can get an output in any size we want.

"""

part1_q3 = r"""
    We don't shuffle the order of batches when training because the meaning of the sequence of the input(that represents a text) is important to the model training.
    If we change and shuffle the order of batches, we will lose the meaning and context of the text, and the model will train no words text and  mistakes.  

"""

part1_q4 = r"""
    1.
    Lowering the temperature using softmax for sampling helps to make the output more deterministic. The softmax function, when applied with a lower temperature, amplifies the probabilities of high-scoring items and suppresses the probabilities of low-scoring items.

    2.
    Very High temperature makes the softmax output become close to a uniform distribution, with nearly equal probabilities for all items. This high temperature leads to increased randomness and diversity in the generated output. the model may result in less coherent or meaningful output as the probabilities are evenly spread across different items.
    

    3.
    Low temperature makes the softmax output become more peaked, with a single item dominating with a probability close to 1.0. In this case, the generated output becomes highly deterministic, relying on the most likely choices at each step. The low temperature constrains the model to generate sequences that closely match the training data, which leads to a lack of creativity and diversity, leading to repetitive or overly predictable sequences.

    


"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
   
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


"""

part2_q2 = r"""
**Your answer:**


"""

part2_q3 = r"""
**Your answer:**



"""

part2_q4 = r"""
**Your answer:**


"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers




part3_q1 = r"""
**Your answer:**

"""

part3_q2 = r"""
**Your answer:**


"""


part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**


"""


# ==============
