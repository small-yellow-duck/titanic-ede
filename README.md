
## Encoder-Decoder-Encoder for tabular data
A major obstacle of using autoencoders for tabular data is that it is not clear how to construct a loss for features with different units - or even where some features are continuous-valued while other features are discrete-valued. The idea behind the encoder-decoder strategy is to evaluate the representation of the items in the *latent space* instead of in the input space as an autoencoder does.

The encoder-decoder algorithm works by encoding an input, `X`, into a vector `E(X)`, decoding the vector back into the input space, `D(E(X)`, then finally re-encoding the result back into a vector, `E(D(E(X)))`. The goal is to minimize the difference between the first encoding, E(X), and the second encoding, `E(D(E(X)))`. At the same time, we want to maximize the distance between the encodings of two different inputs, `E(X1)` and `E(X2)`. Finally, we need a third loss which forces the encoder to also act as a discriminator: by detaching `D(E(X))` from the graph we get an item, `X'` and we try to make the encoder maximize the distance between the encoding `E(X)` and the encoding of the 'faked' item, `E(X')`.

The encoder-decoder-encoder for tabular data is described more thoroughly in the <a href="https://github.com/small-yellow-duck/titanic-ede/blob/master/unsupervised%20deep%20learning%20with%20enc-dec-enc%20-%202019-07-22.pptx">slides</a> in this repo.


#### "Gaussian Overlap: an alternative to the contrastive loss"
The contrastive loss does not pair nicely with variational approaches because the contrastive loss includes a margin term, <img src="https://latex.codecogs.com/svg.latex?m">, which sets the length scale for the latent space. In variational approaches, this length scale is set by a noise term, <img src="https://latex.codecogs.com/svg.latex?\sigma">.

In order to use a variational approach, I propose the Gaussian Overlap loss:

<img src="https://latex.codecogs.com/svg.latex?>\mathcal{L}</img>

<img src="https://latex.codecogs.com/svg.latex?L=-t ln(1- erf(|\mu_i - \mu_j|/2)) +  (1-t)  ln(erf(|\mu_i - \mu_j|/2))">,

<img src="https://latex.codecogs.com/svg.latex?t=0"> if <img src="https://latex.codecogs.com/svg.latex?\mu_i"> and <img src="https://latex.codecogs.com/svg.latex?\mu_j"> are embeddings which are meant to describe the same item and <img src="https://latex.codecogs.com/svg.latex?t=1"> the items are not the same.

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{L} = -t ln(1- erf(|\mu_i - \mu_j|/2)) +  (1-t)  ln(erf(|\mu_i - \mu_j|/2))" />


### Encoding missing categorical data