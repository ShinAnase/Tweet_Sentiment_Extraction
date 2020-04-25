import numpy as np
import torch


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.001):
        #何回スコアの改悪を許すか。(patience数分連続で改悪が行われたら学習をストップさせる。)
        self.patience = patience
        self.counter = 0
        #"min":scoreを下げたいとき。
        #"max":scoreを上げたいとき。
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        #変分。class内に閉じた変数なのでここでしか値をいじれない。
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    #EarlyStoppingのインスタンスが呼び出されたらここを通る。
    def __call__(self, epoch_score, model, model_path):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        #改悪があった場合はカウンタを1増やす。
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            #カウンタが上限(patience)に達したら学習終了。
            if self.counter >= self.patience:
                self.early_stop = True
        #改善が見られたらカウンタを0に初期化。
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    #スコアの改善結果出力&モデルのセーブ。
    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score


def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
