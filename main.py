import time
from collections import Counter
from data import build_corpus
from util1 import CRFModel
from util2 import load_model, save_model, Metrics

CRF_MODEL_PATH = './ckpts/crf.pkl'
REMOVE_O = False  # 在评估的时候是否去除O标记


def crf_train_eval(train_data, test_data, remove_O=False):
    # 训练CRF模型
    train_word_lists, train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data

    crf_model = CRFModel()
    crf_model.train(train_word_lists, train_tag_lists)
    save_model(crf_model, CRF_MODEL_PATH)

    pred_tag_lists = crf_model.test(test_word_lists)

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists


def main():
    """训练模型，评估结果"""
    # 读取数据
    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train")
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    # 训练评估CRF模型
    print("正在训练评估CRF模型...")
    crf_pred = crf_train_eval(
        (train_word_lists, train_tag_lists),
        (test_word_lists, test_tag_lists))

    # 加载并评估CRF模型
    print("加载并评估crf模型...")
    crf_model = load_model(CRF_MODEL_PATH)
    crf_pred = crf_model.test(test_word_lists)
    metrics = Metrics(test_tag_lists, crf_pred, remove_O=REMOVE_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()


if __name__ == "__main__":
    main()