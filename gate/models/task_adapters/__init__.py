from typing import Iterator, Tuple

import torch
import torch.nn as nn


class BaseAdapterModule(nn.Module):
    """
    Base module for an adapter that can include an encoder.
    It supports freezing the encoder and using instance normalization in the stem.

    Attributes:
        encoder (nn.Module): The encoder module.
        freeze_encoder (bool): Flag to freeze the encoder parameters.
        use_stem_instance_norm (bool): Flag to use instance normalization in the stem.
    """

    def __init__(
        self,
        encoder: nn.Module,
        freeze_encoder: bool = False,
        use_stem_instance_norm: bool = False,
    ):
        """
        Initializes the BaseAdapterModule.

        Args:
            encoder (nn.Module): The encoder module.
            freeze_encoder (bool): Whether to freeze the encoder parameters.
            use_stem_instance_norm (bool): Whether to use instance normalization in the stem.
        """
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self.encoder = encoder
        self.use_stem_instance_norm = use_stem_instance_norm

        if self.use_stem_instance_norm:
            self.stem_instance_norm = nn.InstanceNorm2d(
                num_features=3, affine=True
            )

    def encoder_parameters(self) -> Iterator[torch.nn.Parameter]:
        """
        Returns an iterator over the encoder's parameters.

        Returns:
            Iterator[torch.nn.Parameter]: Iterator over the encoder's parameters.
        """
        return self.encoder.parameters()

    def encoder_named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """
        Returns an iterator over the encoder's named parameters.

        Args:
            prefix (str): Prefix for the parameter names.
            recurse (bool): Whether to recurse into submodules.

        Returns:
            Iterator[Tuple[str, torch.nn.Parameter]]: Iterator over the encoder's named parameters.
        """
        return self.encoder.named_parameters(prefix, recurse)

    def adapter_parameters(self) -> Iterator[torch.nn.Parameter]:
        """
        Returns an iterator over the adapter's parameters, excluding the encoder's parameters.

        Returns:
            Iterator[torch.nn.Parameter]: Iterator over the adapter's parameters.
        """
        for name, param in super().named_parameters():
            if not name.startswith("encoder"):
                yield param

    def adapter_named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """
        Returns an iterator over the adapter's named parameters, excluding the encoder's named parameters.

        Args:
            prefix (str): Prefix for the parameter names.
            recurse (bool): Whether to recurse into submodules.

        Returns:
            Iterator[Tuple[str, torch.nn.Parameter]]: Iterator over the adapter's named parameters.
        """
        for name, param in super().named_parameters(prefix, recurse):
            if not name.startswith(f"{prefix}encoder"):
                yield name, param

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        """
        Returns an iterator over the module's parameters. If the encoder is frozen,
        only the adapter's parameters are returned.

        Args:
            recurse (bool): Whether to recurse into submodules.

        Returns:
            Iterator[torch.nn.Parameter]: Iterator over the module's parameters.
        """
        if self.freeze_encoder:
            return self.adapter_parameters()
        return super().parameters(recurse)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """
        Returns an iterator over the module's named parameters. If the encoder is frozen,
        only the adapter's named parameters are returned.

        Args:
            prefix (str): Prefix for the parameter names.
            recurse (bool): Whether to recurse into submodules.

        Returns:
            Iterator[Tuple[str, torch.nn.Parameter]]: Iterator over the module's named parameters.
        """
        if self.freeze_encoder:
            return self.adapter_named_parameters(prefix, recurse)
        return super().named_parameters(prefix, recurse)
