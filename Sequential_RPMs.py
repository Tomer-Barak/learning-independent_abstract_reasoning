import numpy as np
import matplotlib.pyplot as plt
from skimage import draw
from random import shuffle
import matplotlib.image as mpimg
from torch.utils.data import Dataset
import itertools
import torch
from inspect import currentframe, getframeinfo


class shape(list):

    def __init__(self):
        self.append(np.array((  # triangle
            (0.5, 0.0),
            (-0.25, 0.5 * np.sin(2 * np.pi / 3)),
            (-0.25, 0.5 * np.sin(4 * np.pi / 3)),
        )))

        self.append(np.array((  # square
            (0.3535533905932738, 0.3535533905932738),
            (-0.35355339059327373, 0.3535533905932738),
            (-0.35355339059327384, -0.35355339059327373),
            (0.3535533905932737, -0.35355339059327384),
        )))

        self.append(np.array((  # hexagon
            (0.15450849718747373, 0.47552825814757677),
            (-0.40450849718747367, 0.2938926261462366),
            (-0.4045084971874737, -0.2938926261462365),
            (0.15450849718747361, -0.4755282581475768),
            (0.5, -1.2246467991473532e-16),
        )))

        self.append(np.array((  # star
            (0.15450849718747373, 0.47552825814757677),
            (-0.06180339887498947, 0.19021130325903074),
            (-0.40450849718747367, 0.2938926261462366),
            (-0.2, 2.4492935982947065e-17),
            (-0.4045084971874737, -0.2938926261462365),
            (-0.061803398874989514, -0.1902113032590307),
            (0.15450849718747361, -0.4755282581475768),
            (0.16180339887498948, -0.11755705045849468),
            (0.5, -1.2246467991473532e-16),
            (0.1618033988749895, 0.11755705045849459),
        )))

        self.append(0.5)  # circle


def rotate(l, n):
    return l[n:] + l[:n]


def unique_list(list1):
    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        try:
            if x not in unique_list:
                unique_list.append(x)
        except:
            pass
    return unique_list


class seq_RPMs():
    def __init__(self, HP):
        self.rng = np.random.RandomState()
        self._grid_size = HP['grid_size']
        self._seq_length = HP['seq_length']  # the test sequence length including the correct option
        self._num_of_wrong_answers = HP['num_of_wrong_answers']
        self._plot = HP['plot_test']
        self._seq_prop = HP[
            'seq_prop']  # [color,position,size,shape,number] 0:constant, 1:random, 2:predictable, 4:alternating
        self._channels = 3
        self._exponential = HP.get('exponential', False)
        self._sqrt = HP.get('sqrt', False)
        self.HP = HP
        self.initialize()

    def initialize(self):

        self._shapes = shape()

        self._positions = []
        for i in range(3):
            for j in range(3):
                x = int(np.floor((0.5 + i) * self._grid_size / 3))
                y = int(np.floor((0.5 + j) * self._grid_size / 3))
                self._positions.append(np.array((x, y)))

        self._sizes = list(np.linspace(15 / 100 * self._grid_size, self._grid_size * 0.9 / 3, self._seq_length + 1))
        self._sizes = self._sizes[1:]

        self._colors = list((np.array(range(self._seq_length + 1)) + 1) * 1 / (self._seq_length + 1))
        self._colors = self._colors[1:]

        self._numbers = list(np.array(range(9)) + 1)
        self._numbers = self._numbers[1:]

        if self._exponential:
            self._sizes = list(
                np.logspace(np.log10(self._sizes[0]), np.log10(self._sizes[-1]), len(self._sizes), base=10))
            self._colors = list(
                np.logspace(np.log10(self._colors[0]), np.log10(self._colors[-1]), len(self._colors), base=10))
        elif self._sqrt:
            self._sizes = [np.sqrt(i) for i in
                           np.linspace(self._sizes[0] ** 2, self._sizes[-1] ** 2, len(self._sizes))]
            self._colors = [np.sqrt(i) for i in
                            np.linspace(self._colors[0] ** 2, self._colors[-1] ** 2, len(self._colors))]

        self.create_data()

    def create_data(self):

        if self._seq_prop["shape"] == 4:
            shuffle(self._shapes)
            self._rest_of_shapes = self._shapes[2:]
            self._shapes = self._shapes[0:2]
        if isinstance(self._seq_prop["shape"], str):
            self._shapes = [self._shapes[int(self._seq_prop["shape"])]]
        else:
            shuffle(self._shapes)

        shuffle(self._positions)

        if self._seq_prop["number"] == 2:
            self._numbers = rotate(self._numbers, self.rng.randint(0, len(self._numbers) - self._seq_length + 1))
            self.macro = self._numbers.copy()
        elif self._seq_prop["number"] == 4:
            shuffle(self._numbers)
            self._rest_of_numbers = self._numbers[2:]
            self._numbers = self._numbers[:2]
        elif isinstance(self._seq_prop["number"], str):
            self._numbers = [self._numbers[int(self._seq_prop["number"])]]
        else:
            shuffle(self._numbers)

        if self._seq_prop["color"] == 2:
            self.macro = self._colors.copy()
        elif self._seq_prop["color"] == 4:
            shuffle(self._colors)
            self._rest_of_colors = self._colors[2:]
            self._colors = self._colors[:2]
        elif isinstance(self._seq_prop["color"], str):
            self._colors = [self._colors[int(self._seq_prop["color"])]]
        else:
            shuffle(self._colors)

        if self._seq_prop["size"] == 2:
            self.macro = self._sizes.copy()
        elif self._seq_prop["size"] == 4:
            shuffle(self._sizes)
            self._rest_of_sizes = self._sizes[2:]
            self._sizes = self._sizes[:2]
        elif isinstance(self._seq_prop["size"], str):
            self._sizes = [self._sizes[int(self._seq_prop["size"])]]
        else:
            shuffle(self._sizes)

        tiles = []
        wrong_options_atts = []

        for i in range(self._seq_length + self._num_of_wrong_answers):

            tile = np.zeros((self._grid_size, self._grid_size))

            for j in range(self._numbers[0]):
                if type(self._shapes[0]) is not float:
                    rr, cc = draw.polygon(self._positions[j][0] + self._sizes[0] * self._shapes[0][:, 0],
                                          self._positions[j][1] + self._sizes[0] * self._shapes[0][:, 1],
                                          [self._grid_size, self._grid_size])
                    tile[rr, cc] = self._colors[0]
                else:
                    rr, cc = draw.disk((self._positions[j][0], self._positions[j][1]),
                                       self._sizes[0] * self._shapes[0], shape=[self._grid_size, self._grid_size])
                    tile[rr, cc] = self._colors[0]


            tile = np.tile(np.expand_dims(tile, axis=0), (self._channels, 1, 1))

            tiles.append(tile)

            if (i < self._seq_length - 1):  # Sets the feature values for the test sequence + correct option

                if self._seq_prop["shape"] == 1:
                    shuffle(self._shapes)
                elif self._seq_prop["shape"] == 4:
                    self._shapes = rotate(self._shapes, 1)

                if self._seq_prop["position"] == 1:
                    shuffle(self._positions)

                if self._seq_prop["number"] == 1:
                    shuffle(self._numbers)
                elif self._seq_prop["number"] == 2 or self._seq_prop["number"] == 4:
                    self._numbers = rotate(self._numbers, 1)

                if self._seq_prop["color"] == 1:
                    shuffle(self._colors)
                elif self._seq_prop["color"] == 2 or self._seq_prop["color"] == 4:
                    self._colors = rotate(self._colors, 1)

                if self._seq_prop["size"] == 1:
                    shuffle(self._sizes)
                elif self._seq_prop["size"] == 2 or self._seq_prop["size"] == 4:
                    self._sizes = rotate(self._sizes, 1)

            else:  # Sets the feature values for the wrong answers
                duplicated = True
                while duplicated:
                    if self._seq_prop["shape"] == 1:
                        shuffle(self._shapes)
                    elif self._seq_prop["shape"] == 4:
                        if i == self._seq_length - 1:
                            self._shapes = [self._shapes[1]] + self._rest_of_shapes.copy()
                        shuffle(self._shapes)
                    elif self._seq_prop["shape"] == 3:
                        if i == self._seq_length - 1:
                            self._shapes = self._shapes[1:]
                        shuffle(self._shapes)

                    if self._seq_prop["position"] == 1:
                        shuffle(self._positions)

                    if self._seq_prop["number"] == 1:
                        shuffle(self._numbers)
                    elif self._seq_prop["number"] == 2:
                        if i == self._seq_length - 1:
                            self._numbers = self._numbers[1:]
                        shuffle(self._numbers)
                    elif self._seq_prop["number"] == 3:
                        if i == self._seq_length - 1:
                            self._numbers = self._numbers[1:]
                        shuffle(self._numbers)
                    elif self._seq_prop["number"] == 4:
                        if i == self._seq_length - 1:
                            self._numbers = [self._numbers[1]] + self._rest_of_numbers.copy()
                        shuffle(self._numbers)

                    if self._seq_prop["color"] == 1:
                        shuffle(self._colors)
                    elif self._seq_prop["color"] == 2:
                        if i == self._seq_length - 1:
                            self._colors = self._colors[1:]
                        shuffle(self._colors)
                    elif self._seq_prop["color"] == 3:
                        if i == self._seq_length - 1:
                            self._colors = self._colors[1:]
                        shuffle(self._colors)
                    elif self._seq_prop["color"] == 4:
                        if i == self._seq_length - 1:
                            self._colors = [self._colors[1]] + self._rest_of_colors.copy()
                        shuffle(self._colors)

                    if self._seq_prop["size"] == 1:
                        shuffle(self._sizes)
                    elif self._seq_prop["size"] == 2:
                        if i == self._seq_length - 1:
                            self._sizes = self._sizes[1:]
                        shuffle(self._sizes)
                    elif self._seq_prop["size"] == 3:
                        if i == self._seq_length - 1:
                            self._sizes = self._sizes[1:]
                        shuffle(self._sizes)
                    elif self._seq_prop["size"] == 4:
                        if i == self._seq_length - 1:
                            self._sizes = [self._sizes[1]] + self._rest_of_sizes.copy()
                        shuffle(self._sizes)

                    if type(self._shapes[0]) is not float:  # Forbids duplicated wrong answers
                        wrong_options_atts.append(
                            [self._colors[0], self._sizes[0], self._positions[0][0] + self._positions[0][1] / 2,
                             # The positions sum returns a unique value for each grid place
                             self._numbers[0], len(self._shapes[0])])
                    else:
                        wrong_options_atts.append(
                            [self._colors[0], self._sizes[0], self._positions[0][0] ** 2 + self._positions[0][1] / 2,
                             self._numbers[0], 1])
                    if len(unique_list(wrong_options_atts)) < len(wrong_options_atts):
                        wrong_options_atts = wrong_options_atts[:-1]
                        duplicated = True
                    else:
                        duplicated = False

        if self._plot:
            self.plot(tiles)

        res_tiles = np.array(tiles)

        self.data = res_tiles[:self._seq_length - 1]
        self.options = res_tiles[self._seq_length - 1:]

    def plot(self, tiles):
        fig, axes = plt.subplots(2, self._seq_length)
        mix_inds = np.arange(self._seq_length - 1, self._seq_length + self._num_of_wrong_answers).astype(int)
        shuffle(mix_inds)
        mix_inds = [6, 7, 5, 8]
        cmap = 'gray'
        labelpad = 3
        fontsize = 12

        if self._channels > 1:
            tiles = [t[0] for t in tiles]

        for i in range(self._seq_length + self._num_of_wrong_answers):
            if i < self._seq_length - 1:
                axes[0, i].imshow(1 - tiles[i], origin='lower', interpolation='none', vmin=0, vmax=1, cmap=cmap)
                axes[0, i].set_xlabel('Tile ' + str(i + 1), fontsize=fontsize, labelpad=labelpad)
                axes[0, i].set_yticklabels([])
                axes[0, i].set_xticklabels([])
                axes[0, i].set_yticks([])
                axes[0, i].set_xticks([])
                [k.set_linewidth(1) for k in axes[0, i].spines.values()]

            else:
                axes[1, i + 1 - self._seq_length].imshow(1 - tiles[mix_inds[i + 1 - self._seq_length]],
                                                         origin='lower',
                                                         interpolation='none', vmin=0, vmax=1, cmap=cmap)

                axes[1, i + 1 - self._seq_length].set_xlabel('Choice  ' + str(i + 2 - self._seq_length),
                                                             fontsize=fontsize,
                                                             labelpad=labelpad)

                axes[1, i + 1 - self._seq_length].set_yticklabels([])
                axes[1, i + 1 - self._seq_length].set_xticklabels([])
                axes[1, i + 1 - self._seq_length].set_yticks([])
                axes[1, i + 1 - self._seq_length].set_xticks([])
                [k.set_linewidth(1) for k in axes[1, i + 1 - self._seq_length].spines.values()]
        for i in range(self._seq_length - self._num_of_wrong_answers - 1):
            axes[1, self._num_of_wrong_answers + i + 1].axis('off')

        img = mpimg.imread('question_mark.png')
        axes[0, self._seq_length - 1].imshow((img[:, :, 0]) * 255, vmin=0, vmax=255, interpolation='none', cmap=cmap)
        axes[0, self._seq_length - 1].set_xlabel('Tile ' + str(self._seq_length), fontsize=fontsize, labelpad=labelpad)
        axes[0, self._seq_length - 1].set_yticklabels([])
        axes[0, self._seq_length - 1].set_xticklabels([])
        axes[0, self._seq_length - 1].set_yticks([])
        axes[0, self._seq_length - 1].set_xticks([])
        [k.set_linestyle((0, (5, 10))) for k in axes[0, self._seq_length - 1].spines.values()]

        fig.set_size_inches(6.6, 6.76 * 2 / 5)
        plt.subplots_adjust(wspace=-0.05, hspace=0.5)
        plt.tight_layout()
        plt.show()


class AnswersDataSet(Dataset):
    def __init__(self, test_sequence, answers):
        self.pairs = []
        self.answers = answers
        self.test_sequence = test_sequence
        for i in self.answers:
            self.pairs.append([self.test_sequence[-1], i])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


if __name__ == "__main__":
    seq_prop = {
        # 0 - constant ; 1 - random ; 2 - monotonic ; 4 - alternating ; string - set exact feature value
        "color": 2,
        "position": 0,
        "size": '4',
        "shape": 0,
        "number": 0
    }

    HP = {'seq_prop': seq_prop, 'grid_size': 224, 'seq_length': 6, 'num_of_wrong_answers': 3, 'exponential': True,
          'sqrt': False, 'plot_test': True}

    env = seq_RPMs(HP)
