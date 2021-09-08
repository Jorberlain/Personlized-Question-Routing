#随机游走生成器
import os, sys
import networkx as nx
import random
import numpy as np
import math

from collections import Counter
import itertools

class MetaPathGenerator:
    """MetaPathGenerator

    Args:
        dataset     - 要处理的数据集
        length      - 要生成的随机游走的长度
        num_walks   - 从每个节点开始的随机游走的数量
    """

    def __init__(self, dataset, length=100, coverage=10000):
        self._walk_length = length
        self._coverage = coverage
        self._dataset = dataset
        self.G = nx.Graph()

        self.walks = []
        self.pairs = []

        self.initialize()

    def initialize(self):
        """ 初始化图

        用Uq-Q对和Q-Ua对初始化图.

        Args:
            QR_file - 包含Q-R对的输入文件
            QA_file - 包含Q-A对的输入文件

        """

        DATA_DIR = os.getcwd() + "\\data\\parsed\\" + self._dataset + "\\"
        QR_file = DATA_DIR + "Q_R.txt"
        QA_file = DATA_DIR + "Q_A.txt"
        G = self.G
        # Read in Uq-Q pairs
        with open(QR_file, "r") as fin:
            lines = fin.readlines()
            RQ_edge_list = []
            for line in lines:
                unit = line.strip().split()
                RQ_edge_list.append(["Q_" + unit[0],
                                     "R_" + unit[1]])
            G.add_edges_from(RQ_edge_list)
        with open(QA_file, "r") as fin:
            lines = fin.readlines()
            QA_edge_list = []
            for line in lines:
                unit = line.strip().split()
                QA_edge_list.append(["Q_" + unit[0],
                                     "A_" + unit[1]])
            G.add_edges_from(QA_edge_list)

    def get_nodelist(self, type=None):
        """ 获取特定类型或图中所有节点的节点列表

        Args:
            type - 实体的类型.

        Return:
            nodelist - 标注有节点类型的节点列表
        """
        G = self.G

        if not G.number_of_edges() or not G.number_of_nodes():
            sys.exit("Graph should be initialized before get_nodelist()!")

        if not type:
            return list(G.nodes)
        return [node for node in list(G.nodes)
                if node[0] == type]

    def generate_metapaths(self, patterns, alpha):
        """ 生成随机游走

        从图中生成随机游走

        Args:
            meta_pattern - 指引游走生成的模式
            alpha - 重启的概率

        Return:
            walks - 随机游走集
        """
        G = self.G
        num_walks, walk_len = self._coverage, self._walk_length
        rand = random.Random(0)

        print("Generating Meta-paths ...")

        if not G.number_of_edges() or not G.number_of_nodes():
            sys.exit("Graph should be initialized before generate_walks()!")

        walks = []

        for meta_pattern in patterns:  # 通过模式生成
            print("\tNow generating meta-paths from pattern: \"{}\" ..."
                  .format(meta_pattern))
            start_entity_type = meta_pattern[0]
            start_node_list = self.get_nodelist(start_entity_type)
            for cnt in range(num_walks):  
                print("Count={}".format(cnt))
                rand.shuffle(start_node_list)
                total = len(start_node_list)                
                for ind, start_node in enumerate(start_node_list):
                    if ind % 3000 == 0:
                        print("Finished {:.2f}".format(ind/total))

                    walks.append(
                        self.__meta_path_walk(
                            start=start_node,
                            alpha=alpha,
                            pattern=meta_pattern))

        print("Done!")
        self.walks = walks
        return

    def generate_metapaths_2(self):
        """ 生成随机游走

        从图中生成随机游走

        Args:
            meta_pattern - 指引游走生成的模式
            alpha - 重启的概率

        Return:
            walks - 随机游走集
        """
        G = self.G
        num_walks, walk_len = self._coverage, self._walk_length
        rand = random.Random(0)

        print("Generating Meta-paths ...")

        if not G.number_of_edges() or not G.number_of_nodes():
            sys.exit("Graph should be initialized before generate_walks()!")

        walks = []

        print("\tNow generating meta-paths from deepwalk ...")
        start_node_list = self.get_nodelist()
        for cnt in range(num_walks):  # Iterate the node set for cnt times
            print("Count={}".format(cnt))
            rand.shuffle(start_node_list)
            total = len(start_node_list)
            for ind, start_node in enumerate(start_node_list):
                if ind % 3000 == 0:
                    print("Finished {:.2f}".format(ind/total))
                walks.append(
                    self.__random_walk(start=start_node))

        print("Done!")
        self.walks = walks
        return

    def __random_walk(self, start=None):
        """单个随机游走生成器

        Args:
            rand - 生成随机数的随机目标
            start - 起始节点

        Return:
            walk - 生成的单个游走
        """
        G = self.G
        rand = random.Random()
        walk = [start]
        cur_node = start
        while len(walk) <= self._walk_length:
            possible_next_nodes = [neighbor
                                   for neighbor in G.neighbors(cur_node)]
            next_node = rand.choice(possible_next_nodes)
            walk.append(next_node)
            cur_node = next_node

        return " ".join(walk)

    def __meta_path_walk(self, start=None, alpha=0.0, pattern=None):
        """单个随机游走生成器

        根据元路径模式生成单个游走

        Args:
            rand - 生成随机数的随机目标
            start - 起始节点
            alpha - 重启概率
            pattern - (string) 生成游走所根据的模式
            walk_len - (int) 生成游走的长度

        Return:
            walk - 生成的单个游走

        """
        def type_of(node_id):
            return node_id[0]

        rand = random.Random()
        if not pattern:
            sys.exit("Pattern is not specified when generating meta-path walk")

        G = self.G
        n, pat_ind = 1, 1

        walk = [start]

        cur_node = start

        while len(walk) <= self._walk_length or pat_ind != len(pattern):

            pat_ind = pat_ind if pat_ind != len(pattern) else 1

            if rand.random() >= alpha:
                possible_next_node = [neighbor
                                      for neighbor in G.neighbors(cur_node)
                                      if type_of(neighbor) == pattern[pat_ind]]
                next_node = rand.choice(possible_next_node)
            else:
                next_node = walk[0]

            walk.append(next_node)
            cur_node = next_node
            pat_ind += 1

        return " ".join(walk)

    def write_metapaths(self):
        """将元路径写入文件

        Args:
            walks - 通过`generate_walks`生成的游走
        """

        print("Writing Generated Meta-paths to files ...", end=" ")

        DATA_DIR = os.getcwd() + "\\metapath\\"
        OUTPUT = DATA_DIR + self._dataset + "_" \
                 + str(self._coverage) + "_" + str(self._walk_length) + ".txt"
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)
        with open(OUTPUT, "w") as fout:
            for walk in self.walks:
                print("{}".format(walk), file=fout)

        print("Done!")

    def path_to_pairs(self, window_size):
        """将所有元路径转换为节点对

        Args:
            walks - 要转换的游走
            window_size - 滑动窗口尺寸
        Return:
            pairs - 被打乱的节点对语料库数据集
        """
        pairs = []
        if not self.walks:
            sys.exit("Walks haven't been created.")
        for walk in self.walks:
            walk = walk.strip().split(' ')
            for pos, token in enumerate(walk):
                lcontext, rcontext = [], []
                lcontext = walk[pos - window_size: pos] \
                    if pos - window_size >= 0 \
                    else walk[:pos]

                if pos + 1 < len(walk):
                    rcontext = walk[pos + 1: pos + window_size] \
                        if pos + window_size < len(walk) \
                        else walk[pos + 1:]

                context_pairs = [[token, context]
                                 for context in lcontext + rcontext]
                pairs += context_pairs
        np.random.shuffle(pairs)
        self.pairs = pairs
        return

    def write_pairs(self):
        """将所有节点对写入文件
        Args:
            pairs - 语料库
        Return:
        """
        print("Writing Generated Pairs to files ...")
        DATA_DIR = os.getcwd() + "\\corpus\\"
        OUTPUT = DATA_DIR + self._dataset + "_" + \
                 str(self._coverage) + "_" + str(self._walk_length) + ".txt"
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)
        with open(OUTPUT, "w") as fout:
            for pair in self.pairs:
                print("{} {}".format(pair[0], pair[1]), file=fout)
        return

    def down_sample(self):
        """对训练集向下采样
        
        1. 移除所有诸如"A_11 A_11"的节点对
        2. 记录
        """

        pairs = self.pairs
        pairs = [(pair[0], pair[1])
                 for pair in pairs
                 if pair[0] != pair[1]]
        cnt = Counter(pairs)
        down_cnt = [[pair] * math.ceil(math.log(count))
                    for pair, count in cnt.items()]
        self.pairs = list(itertools.chain(*down_cnt))
        np.random.shuffle(self.pairs)

if __name__ == "__main__":
    if len(sys.argv) < 4 + 1:
        print("\t Usage:{} "
              "[name of dataset] [length] [num_walk] [window_size]"
              .format(sys.argv[0], file=sys.stderr))
        sys.exit(1)
    dataset = sys.argv[1]
    length = int(sys.argv[2])
    num_walk = int(sys.argv[3])
    window_size = int(sys.argv[4])
    
    gw = MetaPathGenerator(length=length, coverage=num_walk, dataset=dataset)

    # gw.generate_metapaths(patterns=["AQRQA"], alpha=0)
    gw.generate_metapaths_2()
    gw.path_to_pairs(window_size=window_size)
    gw.down_sample()
    gw.write_metapaths()
    gw.write_pairs()