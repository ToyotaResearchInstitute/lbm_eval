import dataclasses


@dataclasses.dataclass(kw_only=True)
class InitializationBodyConfig:
    # The time at which all geometries associated with the initialization
    # bodies are removed.
    t_remove: float = 0.5

    # Model instance names of the initialization bodies.
    names: list[str] = dataclasses.field(default_factory=list)
