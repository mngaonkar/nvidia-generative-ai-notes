# Parameter Efficient Fine Tuning
Each PEFT method is specified by a PEFTConfig class which stores the types of adapters applicable to the PEFT method, as well as hyperparameters required to initialize these adapter modules.

The following five PEFT methods are currently supported:

- LoRA: LoraPEFTConfig
- QLoRA: QLoraPEFTConfig
- P-Tuning: PtuningPEFTConfig
- Adapters (canonical): CanonicalAdaptersPEFTConfig
- IA3: IA3PEFTConfig

These config classes simplify experimenting with different adapters by allowing easy changes to the config class.


