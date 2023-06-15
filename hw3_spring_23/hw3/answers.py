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
        batch_size=1024,
        seq_len=64,
        h_dim=256,
        n_layers=3,
        dropout=0.4,
        learn_rate=0.001,
        lr_sched_factor=0.01,
        lr_sched_patience=2,
    )
    # ========================
    return hypers


def part1_generation_params():
    start_seq = "What now, my son"
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    
    # ========================
    return start_seq, temperature


part1_q1 = r"""
    we split the corpus into sequences instead of training on the whole text because we can run the model on the text with paralize, as a result the model will run faster.
    Morvoer, if we will use the whole text, the model will overfit on the train set.

"""

part1_q2 = r"""
    This is possible because while the model predict the next char, we use it as the next input of the model.
    In this way, we can get an ouput in any size that we want.

"""

part1_q3 = r"""
    We dont shuffle the order of batches when training, because the meaning of the sequence of the input(that represent a text) is important to the model training.
    If we change and shuffle the order of batches, we will lose the meaning and context of the text, and the model will train no words text and  mistakes.  

"""

part1_q4 = r"""
    1.
    We lower the temperature for sampling because it help us to control the variance of distribution.
    Higher variance leads to more unifrom distribution.

    2.
    Higher temperature lead to samller variance, becasue y/t is close to zero for all the elements as a result the softmax distribution becomes closer to uniform distribution.
    

    3.
    Lower temperature leads to that the softmax distribution becomes more peaked, and the highest-valued elements dominate with higher probabilities.

    


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
