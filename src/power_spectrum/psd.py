"""2D power spectral density: FFT, square, average in rings of constant |k|.

Method and its deviations from the reference post: see README.md.
Entry point: SpectrumBinner. Correctness: selftest().
"""

from __future__ import annotations

import numpy as np


def reference_power_spectrum(image):
    """The post's code, verbatim. Self-test witness only, one image at a time."""
    from scipy import stats

    npix = image.shape[0]
    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image) ** 2

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, npix // 2 + 1, 1.0)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(
        knrm, fourier_amplitudes, statistic="mean", bins=kbins)
    Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
    return kvals, Abins


def top_ring(npix, kmax):
    """Largest binned ring index for a kmax policy. See README, 'where k stops'."""
    if kmax == "complete":
        return npix // 2 - 1
    if kmax == "nyquist":
        return npix // 2
    if kmax == "corner":
        return int(np.floor(np.hypot(npix // 2, npix // 2) + 0.5))
    raise ValueError(f"unknown kmax: {kmax!r}")


class SpectrumBinner:
    """Ring-averaged power spectrum, precomputed for one image size.

    The ring mask depends only on npix, so it is built once and each batch then
    costs one FFT plus a segmented sum.

    kmax selects how far the rings go: 'complete' stops at the last fully
    sampled ring, 'corner' keeps every mode of the sampled square, including
    the angularly incomplete rings of its corners.
    """

    def __init__(self, npix, subtract_mean=True, normalize=True, shell="count",
                 kmax="complete"):
        self.npix = int(npix)
        self.subtract_mean = bool(subtract_mean)
        self.normalize = bool(normalize)
        self.kmax = str(kmax)

        kfreq = np.fft.fftfreq(self.npix) * self.npix
        kfreq2D = np.meshgrid(kfreq, kfreq)
        knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2).ravel()

        self.kbins = np.arange(0.5, top_ring(self.npix, self.kmax) + 1.5, 1.0)
        self.kvals = 0.5 * (self.kbins[1:] + self.kbins[:-1])
        self.nbins = self.kvals.size
        self.shell_area = np.pi * (self.kbins[1:] ** 2 - self.kbins[:-1] ** 2)
        # fftfreq spans -npix/2 .. npix/2-1, so the radius-npix/2 ring is missing
        # its positive side: last fully sampled ring is npix/2 - 1.
        self.k_complete = self.npix // 2 - 1

        # -1 / nbins mean "outside": the k=0 mode, and any ring past kmax.
        b = np.digitize(knrm, self.kbins) - 1
        keep = (b >= 0) & (b < self.nbins)
        cols = b[keep]
        modes = np.flatnonzero(keep)

        order = np.argsort(cols, kind="stable")
        self._modes_sorted = modes[order]
        cols_sorted = cols[order]
        self.counts = np.bincount(cols_sorted, minlength=self.nbins).astype(float)
        self._empty = self.counts == 0
        self._starts = np.searchsorted(cols_sorted, np.arange(self.nbins))

        self.shell = str(shell)
        if self.shell == "area":
            self._shell_w = self.shell_area
        elif self.shell == "count":
            self._shell_w = self.counts
        else:
            raise ValueError(f"unknown shell: {shell!r}")

    def __call__(self, images):
        """(..., npix, npix) -> (..., nbins)."""
        a = np.asarray(images, dtype=np.float64)
        if a.shape[-2:] != (self.npix, self.npix):
            raise ValueError(f"expected (..., {self.npix}, {self.npix}), got {a.shape}")
        lead = a.shape[:-2]
        a = a.reshape(-1, self.npix, self.npix)

        if self.subtract_mean:
            a = a - a.mean(axis=(1, 2), keepdims=True)

        f = np.fft.fft2(a)
        amp = (f.real ** 2 + f.imag ** 2).reshape(a.shape[0], -1)
        if self.normalize:
            amp /= float(self.npix) ** 4

        sums = np.add.reduceat(amp[:, self._modes_sorted], self._starts, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            p = (sums / self.counts) * self._shell_w
        p[:, self._empty] = np.nan
        return p.reshape(*lead, self.nbins)

    def parseval(self, images):
        """(sum_k P(k), spatial variance) -- equal up to ring truncation."""
        p = self(images)
        a = np.asarray(images, dtype=np.float64).reshape(-1, self.npix, self.npix)
        return np.nansum(p.reshape(-1, self.nbins), axis=1), a.var(axis=(1, 2))


def selftest(npix=150, seed=0, verbose=True):
    """Five numerical checks; raises AssertionError on the first failure.

    1. with every deviation switched off, reproduces the post bit for bit;
    2. Parseval: sum_k P(k) equals the spatial variance;
    3. a pure cos(2*pi*n*x/L) lands in ring n with amplitude A^2/2;
    4. a pure CORNER mode, |k| far past Nyquist, lands in its ring just as
       exactly -- those modes are sampled, not aliased;
    5. kmax='corner' keeps every mode, so Parseval holds for any field.
    """
    rng = np.random.default_rng(seed)
    ok = lambda name, val: verbose and print(f"  [ok] {name:.<52s} {val}")

    img = rng.normal(size=(npix, npix))
    kref, pref = reference_power_spectrum(img)
    b_raw = SpectrumBinner(npix, normalize=False, subtract_mean=False,
                           shell="area", kmax="nyquist")
    p_fast = b_raw(img)
    assert np.allclose(kref, b_raw.kvals), "k axis differs from the post"
    rel = np.max(np.abs(p_fast - pref) / np.abs(pref))
    assert rel < 1e-10, f"spectrum != post (max relative error {rel:.2e})"
    ok("identical to the reference implementation (max relative gap)", f"{rel:.2e}")

    # Built in Fourier space: periodic by construction and band-limited, so
    # nothing is lost to the corners the method discards.
    spec = np.zeros((npix, npix), dtype=complex)
    kf = np.fft.fftfreq(npix) * npix
    kx, ky = np.meshgrid(kf, kf)
    band = np.sqrt(kx ** 2 + ky ** 2) <= npix // 4
    spec[band] = rng.normal(size=band.sum()) + 1j * rng.normal(size=band.sum())
    field = np.real(np.fft.ifft2(spec))

    b_none = SpectrumBinner(npix)
    tot, var = b_none.parseval(field)
    rel = abs(tot[0] - var[0]) / var[0]
    assert rel < 1e-12, f"Parseval violated with shell='count' ({rel:.2e})"
    ok("Parseval EXACT with shell='count' (relative gap)", f"{rel:.2e}")

    b_area = SpectrumBinner(npix, shell="area")
    tot, var = b_area.parseval(field)
    rel = abs(tot[0] - var[0]) / var[0]
    assert rel < 1e-2, f"Parseval violated with shell='area' ({rel:.2e})"
    dev = np.abs(b_area.shell_area - b_area.counts) / b_area.counts
    ok("Parseval with the reference ring area (gap)", f"{rel:.2e}")
    ok("  ... biais surface/nb de modes : 1er anneau | RMS",
       f"{-dev[0] * 100:.0f} % | {np.std(b_area.shell_area / b_area.counts - 1) * 100:.1f} %")

    n_mode, amp = 12, 3.0
    x = np.arange(npix) / npix
    pure = amp * np.cos(2 * np.pi * n_mode * x)[None, :] * np.ones((npix, 1))
    p = b_none(pure)
    assert b_none.kvals[int(np.argmax(p))] == n_mode, "pure mode in the wrong ring"
    rel = abs(p.sum() - amp ** 2 / 2) / (amp ** 2 / 2)
    assert rel < 1e-9, f"pure-mode amplitude wrong ({rel:.2e})"
    ok(f"mode pur n={n_mode} : anneau + amplitude A²/2", f"{rel:.2e}")

    b_corner = SpectrumBinner(npix, kmax="corner")
    nx = ny = npix // 2 - 5
    k_corner = int(round(np.hypot(nx, ny)))
    y = np.arange(npix) / npix
    diag = amp * np.cos(2 * np.pi * (nx * x[None, :] + ny * y[:, None]))
    p = b_corner(diag)
    j = int(np.nanargmax(p))
    assert b_corner.kvals[j] == k_corner, "corner mode in the wrong ring"
    rel = abs(np.nansum(p) - amp ** 2 / 2) / (amp ** 2 / 2)
    assert rel < 1e-9, f"corner-mode amplitude wrong ({rel:.2e})"
    ok(f"mode de coin |k|={k_corner} (> Nyquist {npix // 2}) : anneau + A²/2",
       f"{rel:.2e}")

    rough = rng.normal(size=(npix, npix))
    tot, var = b_corner.parseval(rough)
    rel = abs(tot[0] - var[0]) / var[0]
    assert rel < 1e-12, f"kmax='corner' loses power ({rel:.2e})"
    lost = 1.0 - b_none(rough).sum() / b_corner(rough).sum()
    ok("kmax='corner': exact Parseval on an arbitrary field", f"{rel:.2e}")
    ok("  ... power discarded by stopping at the disc (white noise)",
       f"{100 * lost:.1f} %")

    if verbose:
        print("  self-test: the 5 checks pass")
    return True


if __name__ == "__main__":
    print("self-test de psd.py")
    selftest()
