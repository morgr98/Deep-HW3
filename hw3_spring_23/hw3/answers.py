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
        batch_size=256,
        seq_len=64,
        h_dim=512,
        n_layers=3,
        dropout=0.2,
        learn_rate=0.001,
        lr_sched_factor=0.07,
        lr_sched_patience=6,
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
    Moreover, if we will use the whole text, the model will overfit on the train set, and the RNN network will be very deep so it might be caused to a vanishing or exploding gradients.
    Also, if we split the text to small sequence, it can lead that the model will capture long term dependencies in the text.

"""

part1_q2 = r"""
    This is possible because while the model predicts the next char, we use it as the next input of the model, and updating the hidden state according to the current input and the previous hidden state.
    The model maintain in its memory those hidden states that are effected from previous batches that we trained on, and each hidden layer depends on the input and the current hidden state.
    As a result, it can generate memory longer than the sequence length.
"""

part1_q3 = r"""
    We don't shuffle the order of batches when training because the meaning of the sequence of the input(that represents a text) is important to the model training.
    As we explain before, each batch depends on the hidden state that depends from previos bataches.
    If we change and shuffle the order of batches, we will lose the meaning and context of the text, and the model will train no words text and mistakes. 
    
    

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
    hypers = dict(
        batch_size=64,
        h_dim=256,
        z_dim=64,
        x_sigma2=0.01,
        learn_rate=0.0001,
        betas=(0.9, 0.999),
    )
        
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**

The $\sigma^2$ is the variance of the normal distribution of the likelihood 
$p _{\bb{\beta}}(\bb{X} | \bb{Z}=\bb{z}) = \mathcal{N}( \Psi _{\bb{\beta}}(\bb{z}) , \sigma^2 \bb{I} )$.
In VAE, it represents the cariance of the decoder's output distribution, while effects on the reconstruction accuracy and the smoothness of the generated samples.

When the variance is high the model is more flexible in the samples he generates, which can lead to less accuracy but more diversity.
The model generate samples that are more diversity and variation, and will be less similar to the data set.

When the variance is low the model is less flexible and more narrow, and the samples he generates are closer to the samples he trained on, so the accuracy is higher but he get less diversity.
The model try to minimize the reconstruction loss throught generate samples that will be more similar to the data set.

"""

part2_q2 = r"""
**Your answer:**
1. The reconstruction loss is the difference between the original input and the reconstructed output that generated by the decoder of VAE.
   Reconstruction loss is encouraging the VAE to generate samples close to the trained samples.
The KL divergence loss regularizes the latent space by encouraging it to follow a prior distribution (usually a standard normal distribution).
While training the VAE, the encoder can learn to produce the latent variables that deviate from this desired distribution.

2. The KL loss term makes the latent space distribution to be closer to the normal standard distribution, this way we add regularization into the training model.
Therefore we can control how much diverse are our samples from the latent space and how much the loss close to normal distribution.



3. The benefit of the KL loss term's effect is that it promotes a structured latent space, enabling meaningful interpolation and sampling of latent variables during generation.
It encourages the VAE to learn a more disentangled representation of the data, where each dimension of the latent space captures a different aspect or feature, making it easier to manipulate and control the generated outputs.
VAE can learn to capture meaningful features in the latent space and generate diverse and realistic data samples, and prevent overfitting.

"""

part2_q3 = r"""
**Your answer:**
By maximizing P(X), we encourage the VAE to learn the underlying data distribution and generate realistic samples that capture the essential characteristics of the data.
This step serves as a reconstruction objective and is essential for training the VAE to generate meaningful outputs.

"""

part2_q4 = r"""
**Your answer:**

By modeling the log of the latent-space variance, we ensure that the learned representations of the latent space remain unconstrained and can cover a wide range of values. Taking the logarithm transforms the variance from a positive range to the entire real number line, allowing for more flexibility in representation. Additionally, working with the logarithm helps to stabilize the optimization process, preventing numerical instability becasue the variance values can be small, that lead to numerical underflow.
Also, we can notice that if $\sigma^2$ is between 0 to 1, when we operate the log, we get a wide range between [-inf,0], so we can ensure that the encoded distribution covers a broad spectrum of variances.

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
        dropout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    hypers = dict(
        embed_dim = 256, 
        num_heads = 4,
        num_layers = 3,
<<<<<<< HEAD
        hidden_dim = 128,
        window_size = 32,
        droupout = 0.2,
        lr=0.0001,
=======
        hidden_dim = 64,
        window_size = 16,
        dropout = 0.1,
        lr=0.0005,
>>>>>>> b74f5731e0a6509dd348c5dc74be6ee592bd8833
    )
    # ========================
    return hypers




part3_q1 = r"""
**Your answer:**
Every cell in the input tensor is affecting 'window_size' cells in the output of the window_sliding_attention, when we use 2 layers one on top of the other the cell i affect all the [i-window_size/2, i+window_size/2] cells on the first layer and on the second layer they affecting all the cells in 
[i-2*(window_size/2), i+2*(window_size/2)].
So, when we have d layers each cell is affecting all [i-d*(window_size/2), i+d*(window_size/2)], if we think about text learning the first word is the first cell in the input and it affects the d*(window_size/2) word, this way we can have a very large broader context.

"""

part3_q2 = r"""
**Your answer:**
We can use the Global+sliding window, which means we do the same technique as the regular sliding window but we will also have a few global units that attend to all the other units. Let's say for example the first unit is global so we attend between all the other units so it affected and affects all the other units, this way we get more global context, and because we use only a few global values we are getting time complex of O(nw) + O(nK) = O(nw) when K is small constance.

"""


part4_q1 = r"""
**Your answer:**
By comparing the two different fine-tuning training we can see that the second approach when we are tuning all of the model parameters gets a better accuracy and lower loss, we are also seeing that after 1 epoch we get the best results on the test set and during the second epoch the accuracy of the test set is decreasing, but still gets better results than the approach of tuning only the last 2 layers.
We can explain those results because when we are tuning all the parameters in the model, each learning step has more effect on the model towards the minimum loss therefore the model can fine-tune much faster. That is also the reason the second approach tent to overfit more, as we see in the second epoch (the training loss decreasing and the test loss increasing).

This phenomenon is always happening because the phenomenon relies on the connection between faster learning leading to overfit.

"""

part4_q2 = r"""
**Your answer:**
In language model the lower-level layers capture more general linguistic patterns and representations, while the higher-level layers learn task-specific features, that is why we think that freezing all the parameters except some low layers will not be able to fine-tune properly.

"""


# ==============
