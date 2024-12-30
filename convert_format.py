import json


questions = json.load(open('./playground/data/eval/nocaps/nocaps_val_4500_captions.json', "r"))
new_ques_file = './playground/data/eval/nocaps/nocaps_val_4500_captions.jsonl'
write_file = open(new_ques_file, "w")

for item in questions['images']:
    write_file.write(json.dumps({
        "image": item['file_name'],
        "text": "Describe this image in one sentence.",
        "question_id": item['id']
    }) + "\n")

    write_file.flush()
write_file.close()




# # ============================== download image =============================
# import os
# import json
# import requests


# questions = [json.loads(q) for q in open('./playground/data/eval/nocaps/nocaps_val_4500_captions.jsonl', "r")]
# for line in questions:
#     url = line['image']
#     response = requests.get(url)
#     save_path = os.path.join('./playground/data/eval/nocaps/images', url.split('/')[-1])
#     with open(save_path, 'wb') as f:
#         f.write(response.content)
#     print(line['question_id'])
