from river.compose import TransformerUnion


class TimeAwareTransformerUnion(TransformerUnion):
    def learn_one(self, x, t=None):
        for transformer in self.transformers.values():
            transformer.learn_one(x, t)
