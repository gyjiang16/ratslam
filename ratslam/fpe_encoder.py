import cv2
import numpy as np


class FPEEncoder:
    """
    Fractional Power Encoding for scene recognition.

    Encodes a 2D time surface frame into a compact complex vector
    using a Fourier-based codebook.
    """

    def __init__(self, frame_height, frame_width, downsample_factor=8):
        self.ds = int(downsample_factor)
        self.enc_h = int(frame_height) // self.ds
        self.enc_w = int(frame_width) // self.ds
        if self.enc_h <= 0 or self.enc_w <= 0:
            raise ValueError("Invalid encoded size; check frame size and downsample_factor")

        patch_size = (self.enc_h, self.enc_w)
        self.encoding_matrix = self._create_codebook_fourier(patch_size)
        self.E_flat = self.encoding_matrix.reshape(self.enc_h * self.enc_w, -1).T
        self.N = int(self.E_flat.shape[0])

    def _create_codebook_fourier(self, patch_size):
        L = int(patch_size[0])
        M = int(patch_size[1])
        speed_v = np.tile(np.array([2 * np.pi * l / L for l in range(L)]), M)
        speed_h = np.repeat(np.array([2 * np.pi * m / M for m in range(M)]), L)
        vt = np.exp(1j * speed_v)
        ht = np.exp(1j * speed_h)

        vta = np.array([vt ** m for m in np.arange(-L // 2 + 0.5, L // 2)])
        hta = np.array([ht ** m for m in np.arange(-M // 2 + 0.5, M // 2)])
        encoding_matrix = vta[:, None, :] * hta[None, :, :]

        denom = np.linalg.norm(encoding_matrix[0, :])
        if denom > 1e-12:
            encoding_matrix = encoding_matrix / denom
        return encoding_matrix

    def encode(self, time_surface):
        small = cv2.resize(
            time_surface.astype(np.float32),
            (self.enc_w, self.enc_h),
            interpolation=cv2.INTER_AREA,
        )
        s = self.E_flat @ small.ravel()
        norm = np.sqrt(np.sum(np.abs(s) ** 2))
        if norm > 1e-10:
            s = s / norm
        return s

    @staticmethod
    def similarity(s1, s2):
        dot = np.abs(np.real(np.dot(s1, np.conj(s2))))
        n1 = np.sqrt(np.sum(np.abs(s1) ** 2))
        n2 = np.sqrt(np.sum(np.abs(s2) ** 2))
        if n1 < 1e-10 or n2 < 1e-10:
            return 0.0
        return float(dot / (n1 * n2))
