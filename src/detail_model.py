class ImplicitRepresentation:

    def __init__(self, neural_model, detail_model):
        self.neural_model = neural_model
        assert self.neural_model is not None
        self.detail_model = detail_model

    def __call__(self, x):
        if self.detail_model is not None:
            neural_v = self.neural_model(x)
            detail_v = self.detail_model(x.cpu()).to(neural_v.device).reshape((neural_v.shape[0],1))
            return neural_v + detail_v
        return self.neural_model(x)