import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from polygon.utils.moses_utils import OneHotVocab
from polygon.utils.smiles_char_dict import SmilesCharDictionary

def get_vocabulary(data=None, device=None):
    """ Build vocabulary optionally from data 
    """
    if data:
        vocabulary =  OneHotVocab.from_data(data)
    else:
        # if no vocab was provided use a preset character definition
        sd = SmilesCharDictionary()
        vocabulary = OneHotVocab(sd.idx_char.values())
    if device:
        vocabulary.vectors = vocabulary.vectors.to(device)
    return vocabulary

class VAE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # from defaults
        self.q_cell = "gru"
        self.q_bidir = False
        self.q_d_h = 256
        self.q_n_layers = 1
        self.q_dropout = 0.5
        self.d_cell = "gru"
        self.d_n_layers = 3
        self.d_dropout = 0
        self.d_z = 128
        self.d_d_h = 512
        self.freeze_embeddings = False
        self.vocabulary=None
        self.delta = 0.0  # Delta for δ-VAE (rate constraint)
        self.bow_weight = 0.0  # Bag-of-Words auxiliary loss weight

        # overwrite defaults with passed parameters
        self.__dict__.update(kwargs)

        # if we were supplied with a vocabulary use that otherwise
        if self.vocabulary is None:
            self.vocabulary = get_vocabulary()

        # Create SmilesCharDictionary instance once for encoding/decoding
        self._smiles_char_dict = SmilesCharDictionary()

        # Special symbols
        for ss in ('bos', 'eos', 'unk', 'pad'):
            setattr(self, ss, getattr(self.vocabulary, ss))

        # Word embeddings layer
        n_vocab, d_emb = len(self.vocabulary), self.vocabulary.vectors.size(1)
        self.x_emb = nn.Embedding(n_vocab, d_emb, self.pad)
        self.x_emb.weight.data.copy_(self.vocabulary.vectors)

        # if self.device == torch.device("cuda"):
        #     self.x_emb.cuda()


        if self.freeze_embeddings:
            self.x_emb.weight.requires_grad = False

        # Encoder
        if self.q_cell == 'gru':
            self.encoder_rnn = nn.GRU(
                d_emb,
                self.q_d_h,
                num_layers=self.q_n_layers,
                batch_first=True,
                dropout=self.q_dropout if self.q_n_layers > 1 else 0,
                bidirectional=self.q_bidir
            )
        else:
            raise ValueError(
                "Invalid q_cell type, should be one of the ('gru',)"
            )

        q_d_last = self.q_d_h * (2 if self.q_bidir else 1)
        self.q_mu = nn.Linear(q_d_last, self.d_z)
        self.q_logvar = nn.Linear(q_d_last, self.d_z)

        # Decoder
        if self.d_cell == 'gru':
            self.decoder_rnn = nn.GRU(
                d_emb + self.d_z,
                self.d_d_h,
                num_layers=self.d_n_layers,
                batch_first=True,
                dropout=float(self.d_dropout) if self.d_n_layers > 1 else 0
            )
        else:
            raise ValueError(
                "Invalid d_cell type, should be one of the ('gru',)"
            )

        self.decoder_lat = nn.Linear(self.d_z, self.d_d_h)
        self.decoder_fc = nn.Linear(self.d_d_h, n_vocab)

        # Bag-of-Words prediction layer (only if BoW is enabled)
        if self.bow_weight > 0:
            self.bow_fc = nn.Linear(self.d_z, n_vocab)

        # Grouping the model's parameters
        self.encoder = nn.ModuleList([
            self.encoder_rnn,
            self.q_mu,
            self.q_logvar
        ])
        self.decoder = nn.ModuleList([
            self.decoder_lat,
            self.decoder_rnn,
            self.decoder_fc
        ])
        self.vae = nn.ModuleList([
            self.x_emb,
            self.encoder,
            self.decoder
        ])

    @property
    def device(self):
        #return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return next(self.parameters()).device

    def string2tensor(self, string, device='model'):
        # CRITICAL FIX: Encode multi-character tokens (Br→Y, Cl→X, etc.) BEFORE tokenization
        # This prevents 'r' and 'l' from being mapped to <unk> tokens
        encoded_string = self._smiles_char_dict.encode(string)

        ids = self.vocabulary.string2ids(encoded_string, add_bos=True, add_eos=True)
        tensor = torch.tensor(
            ids, dtype=torch.long,
            device=self.device if device == 'model' else device
        )

        return tensor

    def tensor2string(self, tensor):
        ids = tensor.tolist()
        encoded_string = self.vocabulary.ids2string(ids, rem_bos=True, rem_eos=True)

        # CRITICAL FIX: Decode encoded tokens (Y→Br, X→Cl, etc.) back to original SMILES
        string = self._smiles_char_dict.decode(encoded_string)

        return string

    def get_collate_device(self):
        return self.device

    def get_collate_fn(self,):
        device = self.get_collate_device()

        def collate(data):
            data.sort(key=len, reverse=True)
            tensors = [self.string2tensor(string, device=self.device)
                       for string in data]

            return tensors

        return collate

    def forward(self, x):
        """Do the VAE forward step

        :param x: list of tensors of longs, input sentence x
        :return: float, kl term component of loss
        :return: float, recon component of loss
        :return: float, bow (bag-of-words) auxiliary loss
        """

        # Encoder: x -> z, kl_loss
        z, kl_loss = self.forward_encoder(x)

        # Decoder: x, z -> recon_loss
        recon_loss = self.forward_decoder(x, z)

        # Bag-of-Words auxiliary loss (if enabled)
        if self.bow_weight > 0:
            bow_loss = self.forward_bow(x, z)
        else:
            bow_loss = torch.tensor(0.0, device=z.device)

        return kl_loss, recon_loss, bow_loss

    def encode(self, x):
        """   
        Do the VAE forward step to get the latent space of a tensor 

        :param x: list of tensors of longs, input sentence x
        :return: vector representing encoded latent space
        """
        if not isinstance(x,list):
            x = [x]
            
        # Encoder: x -> z, kl_loss
        z, kl_loss, mu = self.forward_encoder(x, return_mu=True)       

        return mu

    def decode(self, z, x=None):
        if x is not None:
            rl, y = self.forward_decoder(x,z,return_y=True)
            xr = y.argmax(2)
            # trim off eos
            #xr = [i[:(i == self.eos).nonzero()[0]] for i in xr]
            # trim off eos
            xrt = []
            for i in xr:
                try:
                    q = i[:(i==self.eos).nonzero()[0]]
                    xrt.append(q)
                except IndexError:
                    xrt.append(i)
            smiles = [self.tensor2string(i_x) for i_x in xrt]
        else:
            smiles = self.sample(z.shape[0], z=z, multinomial=True)
        return smiles

    def get_latent_space_from_smiles(self, smiles):
        """   
        Do the VAE forward step to get the latent space of a smiles 

        :param x: list of tensors of longs, input sentence x
        :return: vector representing encoded latent space
        """

        x = self.string2tensor(smiles)
        # Encoder: x -> z, kl_loss
        z, kl_loss = self.forward_encoder([x])       

        return z[0]

    def forward_encoder(self, x, return_mu=False):
        """Encoder step, emulating z ~ E(x) = q_E(z|x)

        :param x: list of tensors of longs, input sentence x
        :return: (n_batch, d_z) of floats, sample of latent vector z
        :return: float, kl term component of loss
        """

        x = [self.x_emb(i_x) for i_x in x]
        x = nn.utils.rnn.pack_sequence(x, enforce_sorted=False)

        _, h = self.encoder_rnn(x, None)

        h = h[-(1 + int(self.encoder_rnn.bidirectional)):]
        h = torch.cat(h.split(1), dim=-1).squeeze(0)

        mu, logvar = self.q_mu(h), self.q_logvar(h)
        eps = torch.randn_like(mu)
        z = mu + (logvar / 2).exp() * eps

        # Compute raw KL divergence
        total_kl = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()

        # Apply δ-VAE constraint if delta > 0
        # This ensures KL >= delta, preventing posterior collapse
        if self.delta > 0:
            delta_tensor = torch.tensor(self.delta, device=total_kl.device, dtype=total_kl.dtype)
            kl_loss = torch.max(total_kl, delta_tensor)
        else:
            kl_loss = total_kl

        if return_mu:
             return z, kl_loss, mu
        return z, kl_loss
	
    def forward_decoder(self, x, z, return_y=False):
        """Decoder step, emulating x ~ G(z)

        :param x: list of tensors of longs, input sentence x
        :param z: (n_batch, d_z) of floats, latent vector z
        :return: float, recon component of loss
        """

        lengths = [len(i_x) for i_x in x]

        x = nn.utils.rnn.pad_sequence(x, batch_first=True,
                                      padding_value=self.pad)
        x_emb = self.x_emb(x)

        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)
        x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths,
                                                    batch_first=True,
                                                    enforce_sorted=False)

        # ARCHITECTURAL FIX: Initialize decoder hidden state to zeros instead of from z
        # This removes the decoder bypass and forces the decoder to use only the
        # concatenated z at each time step, preventing posterior collapse
        h_0 = torch.zeros(
            self.decoder_rnn.num_layers, z.size(0), self.d_d_h, device=z.device
        )

        output, _ = self.decoder_rnn(x_input, h_0)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.decoder_fc(output)

        recon_loss = F.cross_entropy(
            y[:, :-1].contiguous().view(-1, y.size(-1)),
            x[:, 1:].contiguous().view(-1),
            ignore_index=self.pad
        )
        if return_y:
            return recon_loss,y
        return recon_loss

    def forward_bow(self, x, z):
        """Bag-of-Words auxiliary loss

        Predicts which tokens appear in the molecule from latent code z.
        Forces decoder to use latent information for token content.

        :param x: list of tensors of longs, input sentence x
        :param z: (n_batch, d_z) of floats, latent vector z
        :return: float, bow loss component
        """
        # Backward compatibility: if bow_fc doesn't exist, return 0
        if not hasattr(self, 'bow_fc'):
            return torch.tensor(0.0, device=z.device)

        # Create bag-of-words target: binary vector indicating which tokens appear
        n_batch = len(x)
        n_vocab = len(self.vocabulary)

        # Pad sequences for batch processing
        x_padded = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.pad)

        # Create bow_target and mark all tokens that appear using scatter
        bow_target = torch.zeros(n_batch, n_vocab, device=z.device)
        bow_target.scatter_(1, x_padded, 1.0)  # Mark all tokens in vocabulary

        # Remove special tokens (pad, bos, eos) from the target
        bow_target[:, self.pad] = 0.0
        bow_target[:, self.bos] = 0.0
        bow_target[:, self.eos] = 0.0

        # Predict token presence from latent code
        bow_logits = self.bow_fc(z)  # (n_batch, n_vocab)

        # Binary cross-entropy loss
        bow_loss = F.binary_cross_entropy_with_logits(
            bow_logits, bow_target, reduction='mean'
        )

        return bow_loss

    def sample_z_prior(self, n_batch):
        """Sampling z ~ p(z) = N(0, I)

        :param n_batch: number of batches
        :return: (n_batch, d_z) of floats, sample of latent z
        """

        return torch.randn(n_batch, self.q_mu.out_features,
                           device=self.x_emb.weight.device)

    def perturb_z(self, z, noise_norm, constant_norm=False):
        if noise_norm > 0.0:
            noise_vec = np.random.normal(0, 1, size=z.shape)
            noise_vec = noise_vec / np.linalg.norm(noise_vec)
            if constant_norm:
                return z + (noise_norm * noise_vec)
            else:
                noise_amp = np.random.uniform(
                    0, noise_norm, size=(z.shape[0], 1))
                return z + torch.tensor(noise_amp * noise_vec, dtype=z.dtype)
        else:
            return z

    def sample(self, n_batch, max_len=100, z=None, temp=1.0, multinomial=True, constrained=False):
        """Generating n_batch samples in eval mode (`z` could be
        not on same device)

        :param n_batch: number of sentences to generate
        :param max_len: max len of samples
        :param z: (n_batch, d_z) of floats, latent vector z or None
        :param temp: temperature of softmax
        :param multinomial: if True use multinomial sampling, else argmax
        :param constrained: if True, enforce SMILES syntax constraints
        :return: list of tensors of strings, samples sequence x
        """
        if constrained:
            return self.sample_constrained(n_batch, max_len=max_len, z=z, temp=temp)
        with torch.no_grad():
            if z is None:
                z = self.sample_z_prior(n_batch)
            z = z.to(self.device)
            z_0 = z.unsqueeze(1)

            # Initial values - use zeros for hidden state (consistent with training)
            h = torch.zeros(
                self.decoder_rnn.num_layers, n_batch, self.d_d_h, device=self.device
            )
            w = torch.tensor(self.bos, device=self.device).repeat(n_batch)
            x = torch.tensor([self.pad], device=self.device).repeat(n_batch,
                                                                    max_len)
            x[:, 0] = self.bos
            end_pads = torch.tensor([max_len], device=self.device).repeat(
                n_batch)
            eos_mask = torch.zeros(n_batch, dtype=torch.bool,
                                   device=self.device)

            # Generating cycle
            for i in range(1, max_len):
                x_emb = self.x_emb(w).unsqueeze(1)
                x_input = torch.cat([x_emb, z_0], dim=-1)

                o, h = self.decoder_rnn(x_input, h)
                y = self.decoder_fc(o.squeeze(1))

                # Mask out special tokens (UNK, BOS, PAD) to prevent sampling them
                # Only EOS is allowed as a special token to properly terminate sequences
                y[:, self.unk] = -float('inf')
                y[:, self.bos] = -float('inf')
                y[:, self.pad] = -float('inf')

                y = F.softmax(y / temp, dim=-1)
                if multinomial:
                    w = torch.multinomial(y, 1)[:, 0]
                else:
                    w = torch.argmax(y,1)


                x[~eos_mask, i] = w[~eos_mask]
                i_eos_mask = ~eos_mask & (w == self.eos)
                # new pytorch error with bool vs byte scaler 
                # try to convert it to byte tensor
                test_condition = torch.zeros((w.shape)).bool().to(self.device)
                test_condition = test_condition | (w==self.eos)
                #i_eos_mask = ~eos_mask & test_condition

                end_pads[i_eos_mask] = i + 1
                eos_mask = eos_mask | i_eos_mask

            # Converting `x` to list of tensors
            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, :end_pads[i]])

            return [self.tensor2string(i_x) for i_x in new_x]

    def sample_constrained(self, n_batch, max_len=100, z=None, temp=1.0):
        """Generate samples with SMILES syntax constraints.

        Enforces:
        - Balanced parentheses
        - Balanced ring closures (1-9)
        - Balanced square brackets
        - Forces closure near end of sequence

        Args:
            n_batch: number of samples to generate
            max_len: maximum sequence length
            z: latent vectors (optional)
            temp: softmax temperature

        Returns:
            list of SMILES strings
        """
        # Token indices from SmilesCharDictionary
        OPEN_PAREN = 25   # '('
        CLOSE_PAREN = 24  # ')'
        OPEN_BRACKET = 16  # '['
        CLOSE_BRACKET = 18  # ']'
        RING_TOKENS = {31: 1, 34: 2, 33: 3, 36: 4, 35: 5, 38: 6, 37: 7, 40: 8, 39: 9, 32: 0}
        # Reverse: ring number -> token index
        RING_TO_TOKEN = {v: k for k, v in RING_TOKENS.items()}
        PERCENT_TOKEN = 22  # '%' for ring closures 10-99 (not tracked, so mask it)

        with torch.no_grad():
            if z is None:
                z = self.sample_z_prior(n_batch)
            z = z.to(self.device)
            z_0 = z.unsqueeze(1)

            # Initial values
            h = torch.zeros(
                self.decoder_rnn.num_layers, n_batch, self.d_d_h, device=self.device
            )
            w = torch.tensor(self.bos, device=self.device).repeat(n_batch)
            x = torch.tensor([self.pad], device=self.device).repeat(n_batch, max_len)
            x[:, 0] = self.bos
            end_pads = torch.tensor([max_len], device=self.device).repeat(n_batch)
            eos_mask = torch.zeros(n_batch, dtype=torch.bool, device=self.device)

            # Tracking state for each sample
            paren_count = torch.zeros(n_batch, dtype=torch.long, device=self.device)
            bracket_count = torch.zeros(n_batch, dtype=torch.long, device=self.device)
            # Track open rings: for each sample, a set of open ring numbers
            open_rings = [set() for _ in range(n_batch)]

            for i in range(1, max_len):
                x_emb = self.x_emb(w).unsqueeze(1)
                x_input = torch.cat([x_emb, z_0], dim=-1)

                o, h = self.decoder_rnn(x_input, h)
                y = self.decoder_fc(o.squeeze(1))

                # Mask special tokens
                y[:, self.unk] = -float('inf')
                y[:, self.bos] = -float('inf')
                y[:, self.pad] = -float('inf')

                # Apply syntax constraints
                for b in range(n_batch):
                    if eos_mask[b]:
                        continue

                    # Can't close parentheses if none are open
                    if paren_count[b] == 0:
                        y[b, CLOSE_PAREN] = -float('inf')

                    # Can't close brackets if none are open
                    if bracket_count[b] == 0:
                        y[b, CLOSE_BRACKET] = -float('inf')

                    # Block % token (multi-digit ring closures not tracked)
                    y[b, PERCENT_TOKEN] = -float('inf')

                    # Near end of sequence: force closing open structures
                    remaining = max_len - i - 1
                    open_structures = paren_count[b].item() + bracket_count[b].item() + len(open_rings[b])

                    if remaining <= open_structures + 1:
                        # Build set of allowed tokens (any closing token + EOS)
                        allowed_tokens = {self.eos}

                        if paren_count[b] > 0:
                            allowed_tokens.add(CLOSE_PAREN)

                        if bracket_count[b] > 0:
                            allowed_tokens.add(CLOSE_BRACKET)

                        # Add all ring-closing tokens for open rings
                        for ring_num in open_rings[b]:
                            ring_tok = RING_TO_TOKEN.get(ring_num)
                            if ring_tok is not None:
                                allowed_tokens.add(ring_tok)

                        # Mask out all non-allowed tokens (vectorized)
                        mask = torch.ones(y.size(1), dtype=torch.bool, device=self.device)
                        mask[list(allowed_tokens)] = False
                        y[b, mask] = -float('inf')

                # Sample tokens
                y = F.softmax(y / temp, dim=-1)
                w = torch.multinomial(y, 1)[:, 0]

                # Update state based on sampled tokens
                for b in range(n_batch):
                    if eos_mask[b]:
                        continue

                    tok = w[b].item()

                    if tok == OPEN_PAREN:
                        paren_count[b] += 1
                    elif tok == CLOSE_PAREN:
                        paren_count[b] = max(0, paren_count[b] - 1)
                    elif tok == OPEN_BRACKET:
                        bracket_count[b] += 1
                    elif tok == CLOSE_BRACKET:
                        bracket_count[b] = max(0, bracket_count[b] - 1)
                    elif tok in RING_TOKENS:
                        ring_num = RING_TOKENS[tok]
                        if ring_num in open_rings[b]:
                            open_rings[b].remove(ring_num)  # Close ring
                        else:
                            open_rings[b].add(ring_num)  # Open ring

                x[~eos_mask, i] = w[~eos_mask]
                i_eos_mask = ~eos_mask & (w == self.eos)
                end_pads[i_eos_mask] = i + 1
                eos_mask = eos_mask | i_eos_mask

            # Convert to strings
            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, :end_pads[i]])

            return [self.tensor2string(i_x) for i_x in new_x]
