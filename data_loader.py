#加载数据
import numpy as np
import os, sys

import gensim
import random
import copy

from collections import Counter

data_index = 0
test_index = 0

class DataLoader():
    def __init__(self, dataset, ID,
                 include_content, coverage, length, answer_sample_ratio):
        print("Initializing data_loader ...")
        self.ANS_SAMPLE_SIZE = 5
        self.PAD_LEN = 256
        self.id = ID
        self.dataset = dataset
        self.include_content = include_content
        self.process = True
        self.answer_sample_ratio = answer_sample_ratio

        self.corpus_path =\
            os.getcwd() + "\\corpus\\" + "{}_{}_{}.txt".format(
                self.dataset, str(coverage), str(length))

        self.mpwalks_path =\
            os.getcwd() + "\\metapath\\" + "{}_{}_{}.txt".format(
                self.dataset, str(coverage), str(length))

        self.DATA_DIR = os.getcwd() + "\\data\\parsed\\{}\\".format(self.dataset)

        print("\tLoading dataset ..." + self.corpus_path)
        self.data = self.__read_data()

        print("\tCounting dataset ...")
        self.count = self.__count_dataset()

        print("\tInitializing sample table ...")
        self.sample_table = self.__init_sample_table()

        print("\tLoading word2vec model ...")
        self.w2vmodel = self.__load_word2vec()  # **非常耗时**

        print("\tLoading questions text ...")
        self.question_text = self.__load_question_text()

        print("\tCreating user-index mapping ...")
        self.uid2ind, self.ind2uid = {}, {}
        self.user_count = self.__create_uid_index()

        print("\tLoading rqa ...")
        self.q2r, self.q2acc, self.a2score, self.q2a = {}, {}, {}, {} 
        self.all_aid = []
        self.__load_rqa()

        print("\tCreating qid embeddings map ...")
        self.qid2emb, self.qid2len = {}, {}
        self.__get_question_embeddings()

        print("\tLoading test sets ...")
        self.testset = self.__load_test()

        print("Done - Data Loader!")

    def __read_data(self):
        """
        读取元路径数据集,
            将数据集加载至数据中

        return:
            data  -  元路径数据集
        """
        with open(self.corpus_path, "r") as fin:
            lines = fin.readlines()
            data = [line.strip().split(" ") for line in lines]
            return data

    def __count_dataset(self):
        """
        读取数据集并计算频率

        args:
            data  -  元路径列表
        returns:
            count  - 排序后的列表
        """
        count_dict = {}
        counter = Counter()
        with open(self.mpwalks_path, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.strip().split(" ")
                counter.update(line)
        return counter.most_common()

    def __init_sample_table(self):
        """
        通过p()^(3/4)创建采样表

        return:
            (sample_table)  -  采样表
        """
        count = [ele[1] for ele in self.count]
        pow_freq = np.array(count) ** 0.75
        ratio = pow_freq / sum(pow_freq)
        table_size = 2e7 # todo: what is this???
        count = np.round(ratio * table_size).astype(np.int64)
        sample_table = []

        for i in range(len(self.count)):
            sample_table += [self.count[i][0]] * count[i]
        return np.array(sample_table)

    def get_train_batch(self, batch_size, neg_ratio):
        """
        从元路径中获取批次以供skip-gram训练

        args:
            window_size  -  滑动窗口尺寸. 后续例子中,
                            窗口尺寸为2. "_ _ [ ] _ _"
            batch_size   -  组成一个批次的元路径数量
            neg_ratio    -  负采样和正采样的比例
        return:
            * 所有向量的形式均为"[eny]_[id]"
            upos         -  u向量的位置（一维张量）
            vpos         -  v向量的位置（一维张量）
            npos         -  负样本位置（二维张量）
        """
        data = self.data
        global data_index

        if batch_size + data_index < len(data):
            batch_pairs = data[data_index: data_index + batch_size]
            data_index += batch_size
        else:
            batch_pairs = data[data_index:]
            data_index = 0
            self.process = False

        u, v = zip(*batch_pairs)
        upos = self.__separate_entity(u)
        vpos = self.__separate_entity(v)

        neg_samples = np.random.choice(
            self.sample_table,
            size=int(len(batch_pairs) * neg_ratio))
        npos = self.__separate_entity(neg_samples)
        aqr, accqr = self.get_answer_sample(upos, self.ANS_SAMPLE_SIZE)
        return upos, vpos, npos, aqr, accqr

    def get_test_batch(self, test_prop):
        """
        建立供测试用的批次

        Args:
            test_prop      -  数据集中以供测试的数据的比例，若没有则使用所有数据集
            test_neg_ratio  -  负测试样例的比例，一般为整数

        Returns:
            测试数据列表，形式如下:
                trank_a - 答案列表 (向量)
                rid - 问题提出者id (标量)
                qid - 问题id (标量)
                accaid - 已采纳答案的回答者id (标量)
        """
        total = len(self.testset)
        if test_prop:
            batch_size = int(total * test_prop)
            batch = random.sample(self.testset, batch_size)
        else:
            batch = self.testset
        return batch

    def get_answer_sample(self, upos, sample_size):
        """
        用于排名CNN

        Args:
            upos  -  中心实体列
            vpos  -  上下文实体列
        Return:
            aqr   -  三个列的列表:
                     A为upos, Q为vpos, R为Q
            acc   -  一个列的列表:
                     在相应位置的已采纳答案id
        """
        length = upos.shape[1]

        # R: 0, A: 1, Q: 2
        datalist = []
        acclist = []
        
        HC_times = 3

        for i in range(length):
            if upos[2][i]:
                qid = upos[2][i]
                aids = []
                aids += self.q2a[qid]
                accaid = self.q2acc[qid]
                rid = self.q2r[qid]

                neg_ans = np.random.choice(
                        self.all_aid, replace=False,
                        size=len(aids) * HC_times).tolist()
                
                for x in neg_ans:
                    datalist.append([rid, x, qid])
                acclist += HC_times * aids
                
                aid_samples = aids
                if len(aid_samples) < sample_size:
                    more_ans = np.random.choice(
                            self.all_aid, replace=False,
                            size=sample_size - len(aid_samples))
                    aid_samples += list(more_ans) 


                for x in aid_samples:
                    datalist.append([rid, x, qid])
                    acclist.append(accaid)
        return np.array(datalist), np.array(acclist)

    def __separate_entity(self, entity_seq):
        """
        将列表"a_1 q_2 r_1 q_2"转换为三个向量
            a: 1 0 0 0
            q: 0 2 0 2
            r: 0 0 1 0

        args:
            entity_seq  -  实体序列, type=np.array[(str)]

        return:
            表示上述矩阵的三位矩阵
        """
        D = {"A": 1, "Q": 2, "R": 0}
        sep = np.zeros(shape=(3, len(entity_seq)))
        for index, item in enumerate(entity_seq):
            split = item.split("_")
            ent_type, ent_id = D[split[0]], int(split[1])
            sep[ent_type][index] = ent_id
        return sep.astype(np.int64)

    def __question_len_emb(self, qid):
        """
        给定qid, 返回级联的单词向量

        args:
            qid  -  qid

        return:
            qvec  -  问题的向量, numpy.ndarray
            q_len  -  问题的长度
        """
        q_len = 0
        qvecs = [[0.0] * 300 for _ in range(self.PAD_LEN)]
        if qid:
            question = self.question_text[qid]
            question = [x for x in question.strip().split(" ")
                          if x in self.w2vmodel.vocab]
            if question:
                qvecs = self.w2vmodel[question].tolist()
                q_len = len(question)
                if q_len > self.PAD_LEN:
                    qvecs = qvecs[:self.PAD_LEN]
                else:
                    pad_size = self.PAD_LEN - q_len
                    qvecs += [[0.0] * 300 for _ in range(pad_size)]
        return q_len, qvecs

    def __load_word2vec(self):
        """
        加载word2vec模型， 并返回模型

        Return:
            model  -  一个词典, ("word", [word-vector]),
                      已加载的word2vec模型
        """
        PATH = os.getcwd() + "\\word2vec_model\\" +\
                "GoogleNews-vectors-negative300.bin"
        model = gensim.models.KeyedVectors.load_word2vec_format(
            fname=PATH, binary=True)
        return model

    def __load_question_text(self):
        """
        从数据集中加载问题, "title" + "content",
            构建qid2sen词典

        Return:
            qid2sen  -  qid:question词典
        """

        qcfile = self.DATA_DIR + "Q_content_nsw.txt"
        qtfile = self.DATA_DIR + "Q_title_nsw.txt"

        qid2sen = {}

        with open(qtfile, "r") as fin_t:
            lines = fin_t.readlines()
            for line in lines:
                id, title = line.split(" ", 1)
                qid2sen[int(id)] = title.strip()

        if self.include_content:
            with open(qcfile, "r") as fin_c:
                lines = fin_c.readlines()
                for line in lines:
                    id, content = line.split(" ", 1)
                    if int(id) in qid2sen:
                        qid2sen[int(id)] += " " + content.strip()
                    else:
                        qid2sen[int(id)] = content.strip()

        return qid2sen

    def __create_uid_index(self):
        """
        创建一个uid-index map以及index-uid map

        Return:
            uid2ind  -  用户id到索引词典
            ind2uid  -  索引到用户词典
            len(lines)  -  网络中的用户数量
        """
        uid_file = self.DATA_DIR + "QA_ID.txt"
        self.uid2ind[0] = 0
        self.ind2uid[0] = 0
        with open(uid_file, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                ind, uid = line.strip().split(" ")
                ind, uid = int(ind), int(uid)
                self.uid2ind[uid] = ind
                self.ind2uid[ind] = uid
            print("data_loader: user_count", len(lines))
            return len(lines)

    def __load_rqa(self):
        """
        加载文件

        加载问题至问题提出者ID: self.q2r
            问题至已采纳答案回答者ID: self.q2acc
            问题至回答者ID: self.q2a (list)
        Return:
            无
        """
        QR_input = self.DATA_DIR + "Q_R.txt"
        QACC_input = self.DATA_DIR + "Q_ACC.txt"
        QSC_input = self.DATA_DIR + "A_score.txt" #modified
        QA_input = self.DATA_DIR + "Q_A.txt"

        aid_set = set()

        with open(QR_input, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                Q, R = [int(x) for x in line.strip().split(" ")]
                self.q2r[Q] = R

        with open(QACC_input, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                Q, Acc = [int(x) for x in line.strip().split(" ")]
                self.q2acc[Q] = Acc

        with open(QSC_input, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                A, sc = [int(x) for x in line.strip().split(" ")]
                self.a2score[Q] = sc

        with open(QA_input, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                Q, A = [int(x) for x in line.strip().split(" ")]
                aid_set.add(A)
                if Q not in self.q2a:
                    self.q2a[Q] = [A]
                else:
                    self.q2a[Q].append(A)
        self.all_aid = list(aid_set)

    def __get_question_embeddings(self):
        """
        从qid迅速加载级联的句子向量

        Return:
            qid2emb  -  已加载的map
        """
        for qid in self.question_text.keys():
            qlen, qvecs = self.__question_len_emb(qid)
            self.qid2emb[qid] = qvecs
            self.qid2len[qid] = qlen
        (zero_len, zero_vecs) = self.__question_len_emb(0)
        self.qid2emb[0] = zero_vecs
        self.qid2len[0] = zero_len 

    def __load_test(self):
        """
        将测试集写入内存
        测试文件格式为:
            rid, qid, accid

        Return:
            test  -  测试列表
        """
        test_file = self.DATA_DIR + "test.txt"
        test_set = []
        with open(test_file, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                test_data = [int(x) for x in line.strip().split()]
                rid, qid, accid = test_data[:3]
                aidlist = test_data[3:]
                test_set.append((rid, qid, accid, aidlist))
        return test_set

    def qid2padded_vec(self, qid_list):
        """
        将qid列表转换为一个padded序列.
        序列的长度写入类中

        Args:
            qid_list  -  qid的列表
        Returns:
            padded array  -  padded数组
        """
        qvecs = [self.qid2emb[qid] for qid in qid_list]
        return qvecs
    
    def q2len(self, qid):
        return self.qid2len[qid]

    def qid2vec_length(self, qid_list):
        """
        将qid列表转换为长度的列表
        Args:
            qid_list  -  qid列表
        Returns:
            len array  -  长度列表
        """
        qlens = [self.qid2len[qid] for qid in qid_list]
        return qlens
    
    def q2emb(self, qid):
        return self.qid2emb[qid]

    def uid2index(self, vec):
        """
        用户id表示到用户索引表示

        Args:
            vec  -  要处理的np.array
        Return:
            转换后的numpy array
        """
        def vfind(d, id):
            if id in d:
                return d[id]
            else:
                return random.choice(list(d.values()))
        vfunc = np.vectorize(lambda x: vfind(self.uid2ind, x))
        return vfunc(vec)

    def index2uid(self, vec):
        """
        用户索引表示到用户id表示

        Args:
            vec  -  要处理的np.array
        Return:
            转换后的numpy array
        """
        vfunc = np.vectorize(lambda x: self.ind2uid[x])
        return vfunc(vec)

if __name__ == "__main__":
    test = DataLoader(dataset="3dprinting")
    test.__load_word2vec()