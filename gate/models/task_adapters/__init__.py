from typing import Iterator, Tuple

import torch
import torch.nn as nn

from gate.models.backbones import GATEEncoder


class BaseAdapterModule(nn.Module):
    def __init__(
        self,
        encoder: GATEEncoder,
        freeze_encoder: bool = False,
        use_stem_instance_norm: bool = False,
    ):
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self.encoder = encoder
        self.use_stem_instance_norm = use_stem_instance_norm

        if self.use_stem_instance_norm:
            if self.use_stem_instance_norm:
                self.stem_instance_norm = nn.InstanceNorm2d(
                    num_features=3, affine=True
                )

    def encoder_parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.encoder.parameters()

    def encoder_named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        return self.encoder.named_parameters(prefix, recurse)

    def adapter_parameters(self) -> Iterator[torch.nn.Parameter]:
        # return all parameters except those in self.encoder in an automated way that explicitly excludes the encoder
        for name, param in self.named_parameters():
            if not name.startswith("encoder"):
                yield param

    def adapter_named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        for name, param in self.named_parameters(prefix, recurse):
            if not name.startswith("encoder"):
                yield name, param

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        if self.freeze_encoder:
            # return all parameters except those in self.encoder
            return self.adapter_parameters()

        return super().parameters(recurse)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        if self.freeze_encoder:
            # return all parameters except those in self.encoder in an automated way that explicitly excludes the encoder
            return self.adapter_named_parameters(prefix, recurse)

        return super().named_parameters(prefix, recurse)
