import torch
import numpy as np
import scipy as sc
from torch.utils.data import DataLoader
import torch.optim as optim
import itertools as it
import torch.nn as nn


def Loss(e_t_0, e1):
    loss = 0
    for i in range(len(e_t_0)):
        loss += torch.norm(e_t_0[i] - e1[i]) ** 2
    loss /= len(e_t_0)
    return loss


def test(e_net, t_net, answers, test_sequence, HP, g_net=None):
    e_net.eval()
    t_net.eval()

    options_prediction_error = []

    xs = torch.tensor(np.concatenate((test_sequence, answers)))
    if HP['GPU']:
        xs = xs.cuda().float()
    es = e_net(xs)

    e_last = es[-len(answers) - 1]
    e_t_last = t_net(torch.unsqueeze(e_last, dim=0))

    for answer_index in range(len(answers)):
        if HP['RN']:
            e1 = es[len(test_sequence) + answer_index]
            e0 = e_last
            g_answer = g_net(torch.cat((e0, e1), dim=0))
            L_pred = torch.abs(g_answer - 1)
        else:
            L_pred = torch.norm(e_t_last - es[len(test_sequence) + answer_index]) ** 2

        options_prediction_error.append(L_pred.item())
    answers_prob = sc.special.softmax(-np.array(options_prediction_error))
    return answers_prob, options_prediction_error


def optimization(test_sequence, e_net, t_net, HP):
    g_net = []
    if HP['RN']:
        g_net = torch.nn.Sequential(torch.nn.Linear(2 * 2592, 200), torch.nn.Tanh(), torch.nn.Linear(200, 100),
                                    torch.nn.Tanh(), torch.nn.Linear(100, 50), torch.nn.Tanh(),
                                    torch.nn.Linear(50, 10), torch.nn.Tanh(),
                                    torch.nn.Linear(10, HP['Z_dim']))

        if HP['GPU']:
            g_net = g_net.cuda()
        optimizer = optim.RMSprop(filter(lambda h: h.requires_grad,
                                         it.chain(e_net.parameters(), g_net.parameters())), lr=HP['lr'])
    else:
        if HP['optim'] == 'RMSprop':
            optimizer = optim.RMSprop(
                filter(lambda h: h.requires_grad, it.chain(e_net.parameters(), t_net.parameters())), lr=HP['lr'])
        elif HP['optim'] == 'SGD':
            optimizer = optim.SGD(filter(lambda h: h.requires_grad,
                                         it.chain(e_net.parameters(), t_net.parameters())), lr=HP['lr'])
        elif HP['optim'] == 'Adam':
            optimizer = optim.Adam(filter(lambda h: h.requires_grad,
                                          it.chain(e_net.parameters(), t_net.parameters())), lr=HP['lr'])

    full_answers_prob = []

    for j in range(HP['steps']):

        optimizer.zero_grad()
        e_net.train()
        t_net.train()
        if HP['RN']:
            g_net.train()

        xs = test_sequence
        if HP['GPU']:
            xs = xs.cuda()

        es = e_net(xs)
        e0 = es[:-1]
        e1 = es[1:]

        if HP['RN']:
            gnet_inputs = torch.cat((e0, torch.unsqueeze(e1[-1], dim=0)), dim=0)
            loss = torch.tensor(0)
            for ginput_i in range(len(gnet_inputs)):
                for ginput_j in range(len(gnet_inputs)):
                    cat_input = torch.cat((gnet_inputs[ginput_i], gnet_inputs[ginput_j]), dim=0).unsqueeze(
                        dim=0)
                    if ginput_j == ginput_i + 1:
                        label = torch.tensor(1.)
                    else:
                        label = torch.tensor(0.)
                    loss = loss + torch.square((g_net(cat_input) - label))
            loss = loss / (len(gnet_inputs) ** 2)
        else:
            e_t_0 = t_net(e0)
            loss = Loss(e_t_0, e1)

        loss.backward()
        optimizer.step()

    return e_net, t_net, full_answers_prob, g_net
