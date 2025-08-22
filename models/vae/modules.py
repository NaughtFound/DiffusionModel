import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, n_emb: int, emb_dim: int, beta: float):
        super().__init__()
        self.n_emb = n_emb
        self.emb_dim = emb_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_emb, self.emb_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_emb, 1.0 / self.n_emb)

    def _calc_encodings(self, z: torch.Tensor):
        z_flattened = z.view(-1, self.emb_dim)

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_emb).to(
            z_flattened.device
        )
        min_encodings.scatter_(1, min_encoding_indices, 1)

        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        return min_encoding_indices, min_encodings, perplexity

    def _calc_loss(self, z: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
        q_latent_loss = torch.mean((z_q - z.detach()) ** 2)
        e_latent_loss = torch.mean((z_q.detach() - z) ** 2)
        loss = q_latent_loss + self.beta * e_latent_loss

        return loss

    def forward(self, z: torch.Tensor):
        z = z.permute(0, 2, 3, 1).contiguous()

        min_encoding_indices, min_encodings, perplexity = self._calc_encodings(z)

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        z_q = z + (z_q - z).detach()

        loss = self._calc_loss(z, z_q)

        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return {
            "loss": loss,
            "z_q": z_q,
            "perplexity": perplexity,
            "encodings": min_encodings,
            "encodings_idx": min_encoding_indices,
        }


class ResidualLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        res_h_dim: int,
    ):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=res_h_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=res_h_dim,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

        self.skip_proj = None

        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.skip_proj is not None:
            x = self.skip_proj(x)

        x = x + self.res_block(x)

        return x


class ResidualStack(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        res_h_dim: int,
        n_res_layers: int,
        hidden_dim: int = None,
    ):
        super().__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.Sequential()

        if hidden_dim is None:
            hidden_dim = out_channels

        for _ in range(n_res_layers - 1):
            self.stack.append(
                ResidualLayer(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    res_h_dim=res_h_dim,
                )
            )
            in_channels = hidden_dim

        self.stack.append(
            ResidualLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                res_h_dim=res_h_dim,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stack(x)
        x = F.relu(x)

        return x
