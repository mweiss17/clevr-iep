import torch
import os
import json
from transformers.modeling_bert import BertLayerNorm, BertEmbeddings, BertEncoder, BertConfig, BertPreTrainedModel



base_path = "../clevr-dataset-gen/output/"
scenes = json.load(open(base_path + "CLEVR_scenes.json", "r"))['scenes']

class TextBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self, txt_inds, txt_mask):
        encoder_inputs = self.embeddings(txt_inds)
        attention_mask = txt_mask

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )
        seq_output = encoder_outputs[0]

        return seq_output



for scene in scenes:
    for view_name, view_struct in scene.items():
        print(view_name)
        for object in view_struct['objects']:
            text = object['text']
            body = text['body']
            bert_cfg = BertConfig()

            text_bert = TextBert.from_pretrained(
                'bert-base-uncased', config=bert_cfg
            )
            import pdb; pdb.set_trace()
            text_bert(
                txt_inds=fwd_results['txt_inds'],
                txt_mask=fwd_results['txt_mask']
            )

            import pdb; pdb.set_trace()
            print(text)
        # image = view_struct['image_filename']