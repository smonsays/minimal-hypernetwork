"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import unittest

import flax.linen as nn
import jax
import jax.numpy as jnp

from hypernetwork import Hypernetwork


class HypernetworkTest(unittest.TestCase):
    def test_output_shape(self):
        latent_dim = 3
        input_dim = 7
        output_dim = 5
        batch_dim = 13

        rng = jax.random.PRNGKey(0)
        input = jnp.ones((batch_dim, input_dim))
        latent = jnp.ones((latent_dim))

        hnet = Hypernetwork(
            input_shape=input.shape,
            target_network=nn.Dense(output_dim),
            rng_keys=("dropout", ),
            weight_generator=nn.Dense
        )
        init_rngs = {'params': rng, 'dropout': rng}
        variables = hnet.init(init_rngs, latent, input)
        output = hnet.apply(variables, latent, input, rngs={'dropout': rng})

        assert output.shape == (batch_dim, output_dim)


if __name__ == '__main__':
    unittest.main()
