from bert.tokenization import FullTokenizer
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# See below link to know details of bert model and its I/F
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

sess = tf.Session()

def create_tokenizer_from_hub_module(bert_module):
    """
    hubから取得したmoduleを元にtokenizerを作成します
    """
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

def convert_string_to_bert_input(tokenizer, input_string, max_seq_length=128):
    """
    文字列をBERTで使える形に変換します
    """
    tokens = []
    tokens.append("[CLS]") # 開始文字
    tokens.extend(tokenizer.tokenize(input_string))
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[0 : (max_seq_length - 2)] # 長い場合は削除
    tokens.append("[SEP]") # 終了文字

    # segment_idsは複数文章を繋げる時に境目を示すための値
    # ref. https://github.com/google-research/bert/blob/master/run_classifier.py#L410-L411
    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(tokens) # paddingを無視するためのmaskっぽい

    # 0-padding
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    return np.array([input_ids]), np.array([input_mask]), np.array([segment_ids])

def main(args):
    bert_module =  hub.Module(BERT_MODEL_HUB)
    tokenizer = create_tokenizer_from_hub_module(bert_module)

    string = 'こんにちは、今日の天気はいかがでしょうか？'

    # tokenize
    token = tokenizer.tokenize(string)
    print(token)

    # inputs->outputs
    input_ids, input_mask, segment_ids = convert_string_to_bert_input(tokenizer, string)
    bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids) 
    bert_outputs = bert_module(inputs=bert_inputs, signature="tokens", as_dict=True)
    pooled_output = bert_outputs["pooled_output"]
    sequence_output = bert_outputs["sequence_output"]
    print(pooled_output)
    print(sequence_output)


if __name__ == "__main__":
    tf.app.run()
