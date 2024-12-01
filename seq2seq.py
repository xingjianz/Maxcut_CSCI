import sys

import copy
import time
from typing import List, Union
import numpy as np
import random
import networkx as nx
from util import read_nxgraph
from util import obj_maxcut

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import tqdm
import evaluate
import torch.distributions as distributions


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src length, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src length, batch size, embedding dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # outputs are always from the top hidden layer
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hidden dim]
        # context = [n layers, batch size, hidden dim]
        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, embedding dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [seq length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # seq length and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, hidden dim]
        # hidden = [n layers, batch size, hidden dim]
        # cell = [n layers, batch size, hidden dim]
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch size, output dim]
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert (
            encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio):
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        # input = [batch size]
        for t in range(1, trg_length):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            # output = [batch size, output dim]
            # hidden = [n layers, batch size, hidden dim]
            # cell = [n layers, batch size, hidden dim]
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
            # input = [batch size]
        return outputs


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def seq2seq_annealing(init_temperature: int, num_steps: int, graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
    print('seq2seq_annealing')

    init_solution = [0] * int(graph.number_of_nodes() / 2) + [1] * int(graph.number_of_nodes() / 2)

    start_time = time.time()
    curr_solution = copy.deepcopy(init_solution)
    curr_score = obj_maxcut(curr_solution, graph)
    init_score = curr_score
    scores = []

    input_dim = 2
    output_dim = 2
    encoder_embedding_dim = 256
    decoder_embedding_dim = 256
    hidden_dim = 512
    n_layers = 2
    encoder_dropout = 0.5
    decoder_dropout = 0.5
    batch_size = 50
    advantage_list = []
    stats_losses = []
    stats_advantages = []

    if torch.cuda.is_available() :
        device = torch.device("cuda")
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    encoder = Encoder(
        input_dim,
        encoder_embedding_dim,
        hidden_dim,
        n_layers,
        encoder_dropout,
    )

    decoder = Decoder(
        output_dim,
        decoder_embedding_dim,
        hidden_dim,
        n_layers,
        decoder_dropout,
    )

    model = Seq2Seq(encoder, decoder, device).to(device)
    model.apply(init_weights)
    print(f"The model has {count_parameters(model):,} trainable parameters")
    optimizer = optim.Adam(model.parameters())

    batch_log_probs = []
    batch_advantages = []

    for k in tqdm.tqdm(range(num_steps)):
        # The temperature decreases
        temperature = init_temperature * (1 - (k + 1) / num_steps)
        # temperature = init_temperature * np.exp(-k / (num_steps / 10))

        src = torch.tensor(curr_solution, dtype=torch.long).unsqueeze(1).to(device)  # [seq length, batch size]
        trg = torch.zeros_like(src).to(device)  # dummy target

        # Get outputs from model
        outputs = model(src, trg, teacher_forcing_ratio=0)  # outputs: [seq length, batch size, output_dim]
        outputs = outputs.squeeze(1)  # [seq length, output_dim]

        # Sample new_solution from outputs
        m = distributions.Categorical(logits=outputs)
        new_solution_tensor = m.sample()  # [seq length]
        log_prob = m.log_prob(new_solution_tensor)  # [seq length]
        log_prob_sum = log_prob.sum()

        # Convert new_solution to numpy array
        new_solution = new_solution_tensor.cpu().tolist()

        new_score = obj_maxcut(new_solution, graph)
        scores.append(new_score)

        # Compute advantage
        advantage = new_score - curr_score
        advantage_list.append(advantage)

        batch_log_probs.append(log_prob_sum)
        batch_advantages.append(advantage)

        if (k + 1) % batch_size == 0:
            # Compute loss
            # loss = - log_prob_sum * advantage

            # Backpropagate loss
            advantages = torch.tensor(batch_advantages, dtype=torch.float32).to(device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            log_probs = torch.stack(batch_log_probs)

            loss = - (log_probs * advantages).mean()
            stats_losses.append(loss.item())
            stats_advantages.append(advantages.mean().item())

            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_log_probs = []
            batch_advantages = []

        delta_e = curr_score - new_score
        if delta_e < 0:
            curr_solution = new_solution
            curr_score = new_score
        else:
            prob = np.exp(- delta_e / (temperature + 1e-6))
            if prob > random.random():
                curr_solution = new_solution
                curr_score = new_score

        if (k + 1) % 100 == 0:
            print(f"Step {k + 1}, Loss: {loss.item():.4f}, Advantage: {advantage:.4f}, Score: {curr_score}")

    print(f"score, init_score of seq2seq_annealing: {curr_score}, {init_score}")

    print("scores: ", scores)
    print("losses: ", stats_losses)
    print("advantages: ", stats_advantages)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)


    return curr_score, curr_solution, scores

if __name__ == '__main__':


    # run alg
    # init_solution = list(np.random.randint(0, 2, graph.number_of_nodes()))

    # read data
    #graph = read_nxgraph('./data/syn/syn_50_176.txt')
    graph = read_nxgraph('./data/gset/gset_14.txt')

    init_temperature = 4
    num_steps = 2000

    sa_score, sa_solution, sa_scores = seq2seq_annealing(init_temperature, num_steps, graph)







