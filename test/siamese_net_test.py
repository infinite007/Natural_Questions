import torch
from utils import LSHUtils
from ai.models import SiameseNetwork


class Test:
    def __init__(self, lsh_id):
        self.lsh_utils = LSHUtils(lsh_id)
        self.siamese_net = SiameseNetwork()

    def get_answer(self, query):
        support_set = self.lsh_utils.get_support_set(query,
                                                     self.siamese_net.embedding_model)

        sn_outputs = self.siamese_net.forward([query], support_set)
        which_question = torch.argmax(sn_outputs)
        question_extracted = support_set[which_question]
        return question_extracted, support_set, which_question


if __name__ == "__main__":
    query = input("QUERY>>>")
    test_element = input("TEST ELEMENT>>>")
    test = Test()
    ans, ss, sn_outs = test.get_answer(query)
    print(ans)
    print(ss)
    print(sn_outs)


