"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.flatten_util import ravel_pytree


class Hypernetwork(nn.Module):
    """
    Hypernetwork nn.Module for arbitrary target networks and weight generators.
    """
    input_shape: Tuple[int]
    target_network: nn.Module
    weight_generator: nn.Module
    rng_keys: Tuple[str] = ()

    @staticmethod
    def get_target_apply(
        module: nn.Module, input_shape: Tuple[int], rng_keys: Tuple[str]
    ):

        init_rngs = {key: jax.random.key(0) for key in ("params",) + rng_keys}
        variables = module.init(init_rngs, jnp.empty(input_shape))

        if len(variables) != 1:
            raise NotImplementedError(
                "Not implemented for stateful models (e.g. using batch norm)"
            )

        params_flat, unflatten_fn = ravel_pytree(variables["params"])
        num_params = len(params_flat)

        def apply_fn(params_flat, input, rngs):
            variables = {"params": unflatten_fn(params_flat)}
            return module.apply(variables, input, rngs=rngs)

        return apply_fn, num_params

    def setup(self) -> None:
        self.apply_fn, self.num_params = self.get_target_apply(
            module=self.target_network,
            input_shape=self.input_shape,
            rng_keys=self.rng_keys,
        )
        # NOTE: Assuming weight generator output_dim defined as attribute `features`
        self.weight_generator_ = self.weight_generator(features=self.num_params)

    def __call__(self, latent, input):
        params_flat = self.weight_generator_(latent)
        rngs = {key: self.make_rng(key) for key in self.rng_keys}

        return self.apply_fn(params_flat, input, rngs=rngs)
