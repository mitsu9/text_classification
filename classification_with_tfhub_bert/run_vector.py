from bert.tokenization import FullTokenizer
from bert import modeling # https://github.com/google-research/bert/blob/master/modeling.py
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import collections

# See below link to know details of bert model and its I/F
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

sess = tf.Session()
layer_indexes = [-1, -2, -3, -4]

class InputExample(object):

  def __init__(self, unique_id, text_a, text_b):
    self.unique_id = unique_id
    self.text_a = text_a
    self.text_b = text_b

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids


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

    return input_ids, input_mask, segment_ids

def convert_string_to_bert_input_for_bert_module(tokenizer, input_string, max_seq_length=128):
    input_ids, input_mask, segment_ids = convert_string_to_bert_input(tokenizer, input_string, max_seq_length)
    return np.array([input_ids]), np.array([input_mask]), np.array([segment_ids])

def model_fn(features, labels, mode, params):
    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    input_type_ids = features["input_type_ids"]

    bert_config = modeling.BertConfig.from_json_file("./bert_config.json")

    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=input_type_ids,
        use_one_hot_embeddings=False)

    tvars = tf.trainable_variables()
    init_checkpoint = None
    scaffold_fn = None

    # TODO: 初期化処理っぽいし要らなさそうなので抜いたら動いた. 入れると動かなくなるので要調査.
    # (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    # tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    all_layers = model.get_all_encoder_layers()

    predictions = {
        "unique_id": unique_ids,
    }

    for (i, layer_index) in enumerate(layer_indexes):
        predictions["layer_output_%d" % i] = all_layers[layer_index]

    output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    
    return output_spec

def input_fn_builder(features, seq_length):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_unique_ids = []
    all_input_ids = []
    all_input_mask = []
    all_input_type_ids = []

    for feature in features:
        all_unique_ids.append(feature.unique_id)
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_input_type_ids.append(feature.input_type_ids)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "unique_ids":
                tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_type_ids":
                tf.constant(
                    all_input_type_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
        })

        d = d.batch(batch_size=batch_size, drop_remainder=False)
        return d

    return input_fn

def main(args):
    bert_module =  hub.Module(BERT_MODEL_HUB)
    tokenizer = create_tokenizer_from_hub_module(bert_module)

    string = 'こんにちは、今日の天気はいかがでしょうか？'

    # tokenize
    token = tokenizer.tokenize(string)
    print(token)
    
    # inputs->outputs
    # fine-tuningする時にはここら辺の構造を把握しておく必要がありそう
    input_ids, input_mask, segment_ids = convert_string_to_bert_input_for_bert_module(tokenizer, string)
    bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids) 
    bert_outputs = bert_module(inputs=bert_inputs, signature="tokens", as_dict=True)
    pooled_output = bert_outputs["pooled_output"]
    sequence_output = bert_outputs["sequence_output"]
    print(pooled_output)
    print(sequence_output)

    # create feature vectors
    # ref. https://github.com/google-research/bert/blob/master/extract_features.py
    tokens = tokenizer.tokenize(string)
    input_ids, input_mask, input_type_ids = convert_string_to_bert_input(tokenizer, string)
    feature = InputFeatures(
        unique_id=1,
        tokens=tokens,
        input_ids=input_ids,
        input_mask=input_mask,
        input_type_ids=input_type_ids)
    features = [feature]

    unique_id_to_feature = {}
    unique_id_to_feature[1] = feature

    input_fn = input_fn_builder(features=features, seq_length=128)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        master=None,
        tpu_config=tf.contrib.tpu.TPUConfig(num_shards=8, per_host_input_for_training=is_per_host))

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        predict_batch_size=32)

    for result in estimator.predict(input_fn, yield_single_examples=True):
        unique_id = int(result["unique_id"])
        feature = unique_id_to_feature[unique_id]
        all_features = []
        for (i, token) in enumerate(feature.tokens):
            all_layers = []
            for (j, layer_index) in enumerate(layer_indexes):
                layer_output = result["layer_output_%d" % j]
                layers = collections.OrderedDict()
                layers["index"] = layer_index
                layers["values"] = [
                    round(float(x), 6) for x in layer_output[i:(i + 1)].flat
                ]
                all_layers.append(layers)
            features = collections.OrderedDict()
            features["token"] = token
            features["layers"] = all_layers
            all_features.append(features)
        
        print(f'unique_id: {unique_id}')
        feature = all_features[-1]
        print(f'token: {feature["token"]}')
        print(f'layer_index: {feature["layers"][0]["index"]}')
        print(f'values: {feature["layers"][0]["values"]}')
        print(f'values length: {len(feature["layers"][0]["values"])}')

if __name__ == "__main__":
    tf.app.run()
