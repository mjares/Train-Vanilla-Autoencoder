class AutoencoderResults:
    def __init__(self):
        self.full_outputs = []
        self.full_latent_space = []
        self.loss_log = []
        self.latent_dim = 0
        self.hidden_size = 0
        self.hidden_layers = 0
        self.leaky_slope = 0
        self.notes = ' '
        self.scales = None
