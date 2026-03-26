import configparser
from dataclasses import dataclass

@dataclass
class MetaData:

    # Global data
    geometry_dim: int 
    training_type: str
    gradient_corrected: bool    = False
    detail_field_computed: bool = False


    # Neural model architecture
    architecture_type: str      = None
    n_layers: int               = None
    layer_size: int             = None

    # Neural training parameters
    n_epochs: int               = None
    optimizer: str              = None
    learning_rate: float        = None
    n_sampled_points: int       = None
    resampling_freq: int        = None
    n_resample: int             = None

    # Detail field parameters
    n_primitives: int           = None
    support_size: float         = None
    adaptative_support: bool    = None


    @classmethod
    def load_from_file(cls, file_path: str):
        parser = configparser.ConfigParser()
        parser.read(file_path)
        val_or_none = lambda x,tp: tp(x) if x is not None else None

        return MetaData(
            geometry_dim = int(parser.get("global", "geometry_dim")),
            training_type = parser.get("global", "training_type"),
            gradient_corrected = parser.get("global", "gradient_corrected") == "True",
            detail_field_computed = parser.get("global", "detail_field_computed") == "True",

            architecture_type = parser.get("neural_model", "architecture"),
            n_layers = int(parser.get("neural_model", "n_layers")),
            layer_size = int(parser.get("neural_model", "layer_size")),

            n_epochs = int(parser.get("neural_training", "n_epochs")),
            optimizer = parser.get("neural_training", "optimizer"),
            learning_rate = float(parser.get("neural_training", "learning_rate")),
            n_sampled_points = int(parser.get("neural_training", "n_sampled_points")),
            resampling_freq = int(parser.get("neural_training", "resampling_freq")),
            n_resample = int(parser.get("neural_training", "n_resample")),
            
            n_primitives = val_or_none(parser.get("detail_field", "n_primitives", fallback=None), int),
            support_size = val_or_none(parser.get("detail_field", "support_size", fallback=None), float),
            adaptative_support = parser.get("detail_field", "adaptative_support", fallback=None) == "True",
        )


    def save_to_file(self, file_path: str):
        parser = configparser.ConfigParser()
        parser["global"] = {
            "geometry_dim" : str(self.geometry_dim),
            "training_type" : str(self.training_type),
            "gradient_corrected" : str(self.gradient_corrected),
            "detail_field_computed" : str(self.detail_field_computed),
        }

        parser["neural_model"] = {
            "architecture" : self.architecture_type,
            "n_layers" : str(self.n_layers),
            "layer_size" : str(self.layer_size),
        }

        parser["neural_training"] = {
            "n_epochs" : str(self.n_epochs),
            "optimizer" : self.optimizer,
            "learning_rate" : str(self.learning_rate),
            "n_sampled_points" : str(self.n_sampled_points),
            "resampling_freq" : str(self.resampling_freq),
            "n_resample" : str(self.n_resample)
        }

        if self.detail_field_computed:
            parser["detail_field"] = {
                "n_primitives" : str(self.n_primitives),
                "support_size" : str(self.support_size),
                "adaptative_support" : str(self.adaptative_support),
            }
        else:
            parser["detail_field"] = dict()

        with open(file_path, 'w') as datafile:
            parser.write(datafile)