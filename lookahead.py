import keras
import keras.backend as K


class Lookahead(keras.optimizers.Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        with K.name_scope(self.__class__.__name__):
            self.k = K.constant(k, dtype="int64")
            self.alpha = K.constant(alpha, dtype="float32")
            self.iterations = K.variable(1, dtype="int64", name="iterations")
        self.opt = optimizer

    def get_updates(self, loss, params):
        with K.name_scope(self.__class__.__name__):
            self.slow_weights = [K.variable(p) for p in params]

        opt_updates = self.opt.get_updates(loss, params)
        update_dict = {t.op.inputs[0].name: t.op.inputs[1] for t in opt_updates}
        p_names = [p.name for p in params]
        other_ops = [t for t in opt_updates if t.op.inputs[0].name not in p_names]

        self.updates = [K.update_add(self.iterations, 1)]
        self.updates += other_ops
        
        condition = K.equal(self.iterations % self.k, 0)
        for fast_w, slow_w in zip(params, self.slow_weights):
            self.updates.append(
                K.switch(
                    condition,
                    lambda: K.update(
                        fast_w,
                        K.update_add(
                            slow_w,
                            (K.update(fast_w, update_dict[fast_w.name]) - slow_w)
                            * self.alpha,
                        ),
                    ),
                    lambda: K.update(fast_w, update_dict[fast_w.name]),
                )
            )
        return self.updates

    def get_config(self):
        config = {
            "k": int(K.get_value(self.k)),
            "alpha": float(K.get_value(self.alpha)),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))