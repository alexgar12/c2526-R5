"""
Implementación del modelo ASTGCN (Attention-based Spatio-Temporal Graph
Convolutional Network) para predicción de retrasos en la red de metro de
Nueva York.

Descripción
-----------
ASTGCN extiende STGCN añadiendo mecanismos de atención temporal y espacial
que ponderan dinámicamente los pasos de tiempo y los nodos del grafo antes
de aplicar la convolución de Chebyshev. El modelo se compone de dos bloques
ASTGCNBlock seguidos de una convolución de salida y una capa FC.

Clases exportadas
-----------------
ASTGCN_Metro                    : red completa lista para entrenamiento/inferencia.

Funciones auxiliares exportadas
--------------------------------
calcular_scaled_laplacian        : normaliza el Laplaciano del grafo al rango [-1, 1].
calcular_polinomios_chebyshev    : precalcula los polinomios de Chebyshev T_0..T_{K-1}.

Dependencias
------------
- numpy, torch, torch.nn, torch.nn.functional
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utilidades de grafo
# ---------------------------------------------------------------------------

def calcular_scaled_laplacian(adj_matrix: np.ndarray) -> np.ndarray:
    """
    Calcula el Laplaciano simétrico normalizado escalado al rango [-1, 1].

    El escalado a [-1, 1] es necesario para que los polinomios de Chebyshev
    sean numéricamente estables durante el entrenamiento.

    Parámetros
    ----------
    adj_matrix : np.ndarray (N, N) — matriz de adyacencia ponderada

    Devuelve
    --------
    np.ndarray (N, N) de tipo float32
    """
    adj = adj_matrix.astype(np.float32).copy()
    np.fill_diagonal(adj, 0.0)
    degree = np.sum(adj, axis=1)
    laplacian = np.diag(degree) - adj
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    laplacian_norm = d_mat_inv_sqrt @ laplacian @ d_mat_inv_sqrt
    lambda_max = np.linalg.eigvals(laplacian_norm).real.max()
    if lambda_max == 0 or np.isnan(lambda_max):
        lambda_max = 1.0
    return ((2.0 / lambda_max) * laplacian_norm - np.eye(adj.shape[0], dtype=np.float32)).astype(np.float32)


def calcular_polinomios_chebyshev(scaled_laplacian: np.ndarray, K: int) -> list[torch.Tensor]:
    """
    Precalcula los K primeros polinomios de Chebyshev como tensores PyTorch.

    Usa la recurrencia: T_k = 2*L*T_{k-1} - T_{k-2}, con T_0=I y T_1=L.

    Parámetros
    ----------
    scaled_laplacian : np.ndarray (N, N) — Laplaciano escalado
    K                : int — número de polinomios a calcular

    Devuelve
    --------
    list de K tensores (N, N) de tipo float32
    """
    N = scaled_laplacian.shape[0]
    polys = [np.eye(N, dtype=np.float32)]
    if K > 1:
        polys.append(scaled_laplacian)
    for k in range(2, K):
        polys.append(2 * scaled_laplacian @ polys[k - 1] - polys[k - 2])
    return [torch.tensor(p, dtype=torch.float32) for p in polys]


# ---------------------------------------------------------------------------
# Bloques de atención
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """
    Mecanismo de atención temporal que pondera los pasos de tiempo de la entrada.

    Calcula una matriz de atención (B, T, T) mediante proyecciones lineales de
    query y key, y la usa para reponderar la secuencia temporal de entrada.
    """

    def __init__(self, in_channels: int, num_nodes: int, history_len: int):
        """
        Parámetros
        ----------
        in_channels : número de features por nodo por paso temporal
        num_nodes   : número de nodos en el grafo
        history_len : longitud de la ventana de entrada
        """
        super().__init__()
        self.query_proj = nn.Linear(in_channels, 1, bias=False)
        self.key_proj   = nn.Linear(in_channels, 1, bias=False)
        self.scale      = np.sqrt(max(num_nodes, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parámetros
        ----------
        x : tensor (B, T, N, F)

        Devuelve
        --------
        Matriz de atención (B, T, T) con softmax aplicado
        """
        q = self.query_proj(x).squeeze(-1)                              # (B, T, N)
        k = self.key_proj(x).squeeze(-1)                                # (B, T, N)
        scores = torch.matmul(q, k.transpose(1, 2)) / self.scale        # (B, T, T)
        return torch.softmax(scores, dim=-1)


class SpatialAttention(nn.Module):
    """
    Mecanismo de atención espacial que pondera los nodos del grafo.

    Calcula una matriz de atención (B, N, N) que modula la convolución
    de Chebyshev para enfocarse en los nodos más relevantes.
    """

    def __init__(self, in_channels: int, num_nodes: int, history_len: int):
        """
        Parámetros
        ----------
        in_channels : número de features por nodo por paso temporal
        num_nodes   : número de nodos en el grafo
        history_len : longitud de la ventana de entrada
        """
        super().__init__()
        self.query_proj = nn.Linear(in_channels, 1, bias=False)
        self.key_proj   = nn.Linear(in_channels, 1, bias=False)
        self.scale      = np.sqrt(max(history_len, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parámetros
        ----------
        x : tensor (B, T, N, F)

        Devuelve
        --------
        Matriz de atención espacial (B, N, N) con softmax aplicado
        """
        q = self.query_proj(x).squeeze(-1).transpose(1, 2)             # (B, N, T)
        k = self.key_proj(x).squeeze(-1).transpose(1, 2)               # (B, N, T)
        scores = torch.matmul(q, k.transpose(1, 2)) / self.scale        # (B, N, N)
        return torch.softmax(scores, dim=-1)


class ChebConvWithSpatialAttention(nn.Module):
    """
    Convolución espectral de Chebyshev modulada por la atención espacial.

    Aplica los K polinomios de Chebyshev ponderados por la matriz de atención
    espacial y suma la contribución de cada polinomio con parámetros Theta_k.
    """

    def __init__(self, K: int, cheb_polynomials: list[torch.Tensor], in_channels: int, out_channels: int):
        """
        Parámetros
        ----------
        K                : orden de los polinomios de Chebyshev
        cheb_polynomials : lista de K tensores (N, N) precalculados
        in_channels      : dimensión de entrada
        out_channels     : dimensión de salida
        """
        super().__init__()
        self.K           = K
        self.out_channels = out_channels
        self.Theta = nn.ParameterList([
            nn.Parameter(torch.empty(in_channels, out_channels)) for _ in range(K)
        ])
        for theta in self.Theta:
            nn.init.xavier_uniform_(theta)
        # Polinomios apilados como buffer no entrenable para eficiencia
        self.register_buffer('cheb_polynomials', torch.stack(cheb_polynomials, dim=0))

    def forward(self, x: torch.Tensor, spatial_attention: torch.Tensor) -> torch.Tensor:
        """
        Parámetros
        ----------
        x                : tensor (B, T, N, F)
        spatial_attention: tensor (B, N, N)

        Devuelve
        --------
        Tensor (B, T, N, out_channels) tras aplicar ReLU
        """
        B, T, N, _ = x.shape
        outputs = []
        for t in range(T):
            graph_signal = x[:, t, :, :]                                # (B, N, F)
            output_t = torch.zeros((B, N, self.out_channels), device=x.device, dtype=x.dtype)
            for k in range(self.K):
                T_k    = self.cheb_polynomials[k]                       # (N, N)
                T_k_at = T_k.unsqueeze(0) * spatial_attention           # (B, N, N)
                rhs    = torch.einsum('bij,bjf->bif', T_k_at, graph_signal)
                output_t = output_t + torch.matmul(rhs, self.Theta[k])
            outputs.append(output_t.unsqueeze(1))
        return F.relu(torch.cat(outputs, dim=1))                        # (B, T, N, out_ch)


class ASTGCNBlock(nn.Module):
    """
    Bloque ASTGCN: atención temporal → atención espacial → ChebConv → conv temporal → residual.

    Combina los mecanismos de atención temporal y espacial con la convolución
    de Chebyshev y una convolución temporal de refinamiento. Incluye conexión
    residual y normalización por capas.
    """

    def __init__(
        self,
        in_channels: int,
        K: int,
        cheb_polynomials: list[torch.Tensor],
        num_nodes: int,
        history_len: int,
        out_channels: int,
        temporal_kernel: int = 3,
    ):
        """
        Parámetros
        ----------
        in_channels     : canales de entrada
        K               : orden de Chebyshev
        cheb_polynomials: polinomios precalculados
        num_nodes       : número de nodos
        history_len     : longitud de la ventana
        out_channels    : canales de salida
        temporal_kernel : tamaño del kernel de la convolución temporal
        """
        super().__init__()
        self.temporal_attention = TemporalAttention(in_channels, num_nodes, history_len)
        self.spatial_attention  = SpatialAttention(in_channels, num_nodes, history_len)
        self.cheb_conv          = ChebConvWithSpatialAttention(K, cheb_polynomials, in_channels, out_channels)
        self.time_conv          = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(temporal_kernel, 1), padding=(temporal_kernel // 2, 0)
        )
        self.residual_conv      = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.layer_norm         = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parámetros
        ----------
        x : tensor (B, T, N, F)

        Devuelve
        --------
        Tensor (B, T, N, out_channels) normalizado
        """
        temporal_attention = self.temporal_attention(x)
        x_ta  = torch.einsum('bts,bsnf->btnf', temporal_attention, x)
        spatial_attention  = self.spatial_attention(x_ta)
        x_gc  = self.cheb_conv(x_ta, spatial_attention)
        x_tc  = self.time_conv(x_gc.permute(0, 3, 1, 2))
        res   = self.residual_conv(x.permute(0, 3, 1, 2))
        x_out = F.relu(x_tc + res).permute(0, 2, 3, 1)
        return self.layer_norm(x_out)


# ---------------------------------------------------------------------------
# Modelo completo
# ---------------------------------------------------------------------------

class ASTGCN_Metro(nn.Module):
    """
    ASTGCN con dos bloques de atención espacio-temporal para predicción de
    retrasos en estaciones de metro.

    Parámetros
    ----------
    num_nodes        : número de nodos
    num_features     : features por nodo por paso temporal
    num_targets      : salidas por nodo
    history_len      : longitud de la ventana de entrada
    cheb_polynomials : lista de K tensores (N, N)
    K                : orden de Chebyshev
    hidden_channels  : canales en los bloques ASTGCN
    dropout          : tasa de dropout
    """

    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        num_targets: int,
        history_len: int,
        cheb_polynomials: list[torch.Tensor],
        K: int,
        hidden_channels: int,
        dropout: float,
    ):
        super().__init__()
        self.block1      = ASTGCNBlock(num_features,    K, cheb_polynomials, num_nodes, history_len, hidden_channels)
        self.block2      = ASTGCNBlock(hidden_channels, K, cheb_polynomials, num_nodes, history_len, hidden_channels)
        self.dropout     = nn.Dropout(dropout)
        self.final_conv  = nn.Conv2d(history_len, 1, kernel_size=(1, hidden_channels))
        self.fc          = nn.Linear(num_nodes, num_nodes * num_targets)
        self.num_nodes   = num_nodes
        self.num_targets = num_targets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parámetros
        ----------
        x : tensor (B, T, N, F)

        Devuelve
        --------
        Tensor (B, N, num_targets) con las predicciones por nodo y horizonte
        """
        x = self.block1(x)
        x = self.dropout(x)
        x = self.block2(x)
        x = self.dropout(x)
        x = self.final_conv(x)                  # (B, 1, N, 1) via permuta interna
        x = x.squeeze(-1).squeeze(1)            # (B, N)
        x = self.fc(x)                          # (B, N * num_targets)
        return x.view(-1, self.num_nodes, self.num_targets)
