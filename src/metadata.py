import configparser
from dataclasses import dataclass

@dataclass
class MetaData:
    geometry_dim: int

    # Description of the neural network
    architecture_type: str      = None
    n_layers: int               = None
    layer_size: int             = None

    # Neural training parameters
    n_epochs: int               = None
    optimizer: str              = None
    learning_rate: float        = None
    batch_size: int             = None
    test_batch_size: int        = None
    n_sampled_points: int       = None


    # Detail field parameters
    has_detail_field: bool      = False
    n_centers: int              = None
    sigma: float                = None
    support_size: float         = None
    rbf_shape_id: int           = None
    implicit: bool              = None

    def save_to_file(self, file_path: str):
        parser = configparser.ConfigParser()
        parser["global"] = {
            "geometry_dim" : str(self.geometry_dim),
        }

        parser["neural_model"] = {
            "architecture_type" : self.architecture_type,
            "n_layers" : str(self.n_layers),
            "layer_size" : str(self.layer_size),
        }

        parser["neural_training"] = {
            "n_epochs"         : self.n_epochs,
            "optimizer"        : self.optimizer,
            "learning_rate"    : self.learning_rate,
            "batch_size"       : self.batch_size,
            "test_batch_size"  : self.test_batch_size,
            "n_sampled_points" : self.n_sampled_points,
        }

        if self.has_detail_field:
            parser["detail_field"] = {
                "has_detail_field" : True,
                "n_centers" : self.n_centers,
                "sigma" : self.sigma,
                "support_size" : self.support_size,
                "rbf_shape_id" : self.rbf_shape_id,
                "implicit" : self.implicit
            }
        else:
            parser["detail_field"] = {
                "has_detail_field" : False
            }

        with open(file_path, 'w') as datafile:
            parser.write(datafile)

    @classmethod
    def load_from_file(cls, file_path: str):
        parser = configparser.ConfigParser()
        parser.read(file_path)

        neural_model = parser["neural_model"]
        neural_training = parser["neural_training"]
        detail_field = parser["detail_field"]

        return MetaData(
            geometry_dim = parser["global"].getint("geometry_dim"),

            architecture_type = neural_model["architecture_type"],
            n_layers = neural_model.getint("n_layers"),
            layer_size = neural_model.getint("layer_size"),

            n_epochs = neural_training.getint("n_epochs"),
            optimizer = neural_training["optimizer"],
            learning_rate = neural_training.getfloat("learning_rate"),
            n_sampled_points = neural_training.getint("n_sampled_points"),
            batch_size = neural_training.getint("n_sampled_points"),
            test_batch_size = neural_training.getint("n_sampled_points"),

            has_detail_field = detail_field.getboolean("has_detail_field"),
            n_centers = detail_field.getint("n_centers", fallback=None),
            sigma = detail_field.getfloat("sigma", fallback=None),
            support_size = detail_field.getfloat("support_size", fallback=None),
            rbf_shape_id = detail_field.getint("adaptative_support", fallback=None),
            implicit= detail_field.getboolean("implicit", fallback=None)
        )


    