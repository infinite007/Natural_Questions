import json, re
from pprint import pprint
from more_itertools import padded
from bs4 import BeautifulSoup as bs
from nltk import clean_html


def change_offset(dt, start, end):
    context = []
    start_diff = 0
    end_diff = 0
    clean = re.compile("<.*?>")
    for i, token in enumerate(dt):
        if re.sub(clean, '', token) != '':
            context.append(token)
        elif i < start:
            start_diff += 1
            end_diff += 1
        elif i < end:
            end_diff += 1
    new_start = start - start_diff
    new_end = end - end_diff
    return context, new_start, new_end

x = []
annotations = []
with open("/home/infinite007/Extras/ASU_COURSES/SEM1/CSE576/Natural_Questions/data/nq-train.jsonl", "r") as f:
    for idx, line in enumerate(f):
        if idx == 10:
            break
        obj = json.loads(line)
        # pprint(obj)
        annotation = obj["annotations"]
        if annotation:
            annotation = annotation[0]
            long_answer = annotation["long_answer"]
            short_answers = annotation["short_answers"]
            long_answer_strt_token = long_answer["start_token"]
            long_answer_end_token = long_answer["end_token"]
            if long_answer_strt_token < 0 or long_answer_end_token < -1:
                continue
            # print("question : ", obj["question_text"])
            # print("answer : ", " ".join(obj["document_text"].split()[long_answer_strt_token:long_answer_end_token]))
            if short_answers:
                short_answer_strt_token = long_answer["start_token"]
                short_answer_end_token = long_answer["end_token"]
        else:
            continue
        annotations.append((long_answer_strt_token, long_answer_end_token))
        x.append(obj["document_text"])
print(annotations)
# split_string = "<P>|</P>|<Td>|</Td>|<Tr>|</Tr>|<Th>|</Th>|<Table>|</Table>|<Li>|</Li>"
# documents_split = [re.split(split_string, d) for d in x]
y = bs(x[0], "html.parser").text
actual_page = x[0].split()
# print(actual_page)
# print(y)
split_1 = [i.split() for i in re.split("\s{2,}", y)]
# print(split_1)
offset_map = []

modified_row = 0
prev_modified_row = modified_row
modified_col = -1
prev_modified_col = modified_col
previous_token = ""

# for idx, j in enumerate(actual_page):
#     if (j.startswith("<") and j.endswith(">")) and not (previous_token.startswith("<") and previous_token.endswith(">")):
#         if modified_col == prev_modified_col and (modified_row + 1 - prev_modified_row) == 1:
#             modified_row += 1
#             modified_col = -1
#     modified_col += 1
#     if 150 < modified_row:
#         print("idx : ", idx, modified_row, modified_col)
#         prev_modified_row = modified_row
#         prev_modified_col = modified_col
#     offset_map.append((modified_row, modified_col))
#
# for x_ele, a_ele in zip(x, annotations):
#     obj = change_offset(x_ele.split(), a_ele[0], a_ele[1])
#     print(x_ele.split()[a_ele[0]:a_ele[1]])
#     print(obj[0][obj[1]:obj[2]])
#     print(bs(x_ele, "html.parser").text.split()[obj[1]:obj[2]])



# print(actual_page[1000:])
# print(split_1[150:])
# print(offset_map[-1])
# print(offset_map)
# print()
# for i, j in offset_map:
#     try:
#         print(split_1[i][j])
#     except:
#         print("could find : %d" % i)

# start_mask = 0
# for j_idx, j in enumerate(split_1):
#     for i_idx, i in enumerate(j):
#         for actual_idx, s in enumerate(actual_page[start_mask:]):
#             if s == i:
#                 start_mask = actual_idx
#                 print(j_idx, i_idx, actual_idx)



















# documents_text_split = [[i.split() for i in ds] for ds in documents_split]
# print([len(i) for i in documents_text_split[0]])
# print(documents_text_split[0][0])
# print(documents_text_split[-1])
# print([[len(i) for i in ds] for ds in documents_text_split])
# print([len(ds) for ds in documents_text_split])
# pprint(documents_text_split)
# pprint([[i.split() for i in ds] for ds in documents_split])
# print([len(ds) for ds in documents_split])
