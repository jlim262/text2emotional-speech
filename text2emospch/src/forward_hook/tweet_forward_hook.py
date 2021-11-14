
import logging

from built.forward_hook import ForwardHookBase
from built.registry import Registry


@Registry.register(category='hooks')
class TweetForwardHook(ForwardHookBase):
    def forward(self, inputs, model, is_train):
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs['token_type_ids']
        )

        return outputs
