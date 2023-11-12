import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import Sequential_RPMs as env
import training


class Z_conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.classifier = nn.Identity()

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = torch.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 4)
        x = F.max_pool2d(F.relu(self.conv3(x)), 6)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


class T(nn.Module):
    def __init__(self, HP):
        super(T, self).__init__()
        self.HP = HP
        self.const1 = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        x = x.view(-1, self.HP['Z_dim'])
        x_in = x
        x = self.const1(torch.ones_like(x_in)) + x_in
        return x


def create_net(HP):
    e_net = Z_conv()
    if HP['freeze_conv']:
        for param in e_net.parameters():
            param.requires_grad = False

    if not HP['RN']:
        e_net.classifier = torch.nn.Sequential(torch.nn.Linear(2592, 200), torch.nn.Tanh(), torch.nn.Linear(200, 100),
                                               torch.nn.Tanh(), torch.nn.Linear(100, 50), torch.nn.Tanh(),
                                               torch.nn.Linear(50, 10), torch.nn.Tanh(),
                                               torch.nn.Linear(10, 1))

    if HP['freeze_FC']:
        for param in e_net.classifier.parameters():
            param.requires_grad = False

    t_net = T(HP)
    if HP['freeze_T']:
        for param in t_net.parameters():
            param.requires_grad = False

    if HP['GPU']:
        e_net = e_net.to('cuda')
        t_net = t_net.to('cuda')

    return e_net, t_net


def create_test(HP):
    seq_RPM = env.seq_RPMs(HP)
    dataset = env.AnswersDataSet(seq_RPM.data, seq_RPM.options)
    return torch.from_numpy(dataset.test_sequence).float(), torch.from_numpy(dataset.answers).float()


def run_single_test(HP):
    # Creates the test and answers
    test_sequence, answers = create_test(HP)

    # Creates networks
    z_net, t_net = create_net(HP)

    # Optimization process
    (z_net, t_net, full_answers_prob, g_net) = training.optimization(test_sequence, z_net, t_net, HP)

    # Choosing an answer
    answers_prob, e_options = training.test(z_net, t_net, answers, test_sequence, HP, g_net=g_net)
    answers_prob = np.array(answers_prob)
    if True in np.isnan(answers_prob):
        winner_index = int(np.random.randint(4))
    else:
        winner_index = int(np.argmax(answers_prob))
        chosen_value = answers_prob[winner_index]
        winner_index = int(np.random.choice(np.where(answers_prob == chosen_value)[0]))

    return winner_index


def run_multiple_tests(HP):
    chosen_options = np.zeros((HP['N_tests'], HP['num_of_wrong_answers'] + 1))

    for i in range(HP['N_tests']):
        answer_idx = run_single_test(HP)

        if type(answer_idx) == int:
            chosen_options[i, answer_idx] = 1

        print('Test number', i + 1, flush=True)
        success_rate = np.sum(chosen_options[:, 0]) / (i + 1)
        print('Current success rate:', np.round(success_rate, 3), flush=True)

    total_success_rate = np.mean(chosen_options, axis=0)[0]

    torch.cuda.empty_cache()

    return total_success_rate


if __name__ == "__main__":
    HP = {'GPU': True,

          # Optimization properties
          'steps': 10, 'lr': 1e-5, 'optim': 'RMSprop', 'Z_dim': 1, 'RN': False,
          'freeze_conv': True, 'freeze_T': False, 'freeze_FC': True,

          # Test properties
          'grid_size': 224, 'seq_length': 6, 'num_of_wrong_answers': 3,
          'plot_test': True, 'N_tests': 5, 'exponential': False, 'sqrt': False,

          # Feature properties: 0-constant, 1-random, 2-predictive, 4-alternating, string-exact feature value
          'seq_prop': {"color": 2, "position": 0, "size": '4', "shape": 0, "number": 1}}

    run_multiple_tests(HP)
