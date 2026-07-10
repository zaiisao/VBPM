# ELBO derivation — rendered preview / working draft

> **How to view this rendered:** open this file in VSCode and press `Ctrl/Cmd+Shift+V`
> (Markdown: Open Preview). VSCode renders `$…$` / `$$…$$` with KaTeX out of the box.
> No copy-to-Gemini needed.

**Convention (this draft):** conditioning sets are written out in full — `z_{1:T}`, `b_{1:T}`,
`h_{1:T}` are never abbreviated to bare letters. The encoder `q` conditions on `(b_{1:T}, h_{1:T})`;
the prior `p` on `h_{1:T}` only; the emission on `z_t` alone. Symbols follow the modified notebook
($\dot\phi_t$ = log-tempo, $\phi_t$ = bar phase, $m_t$ = meter) — swap to $s_t,\varphi_t$ if you
prefer the original's names; it changes nothing in the derivation.

---

## 1. The ELBO

$$\log p(b_{1:T}\mid h_{1:T}) = \log \int p(b_{1:T}, z_{1:T}\mid h_{1:T})\,dz_{1:T}$$

$$= \log \int q(z_{1:T}\mid b_{1:T}, h_{1:T})\,\frac{p(b_{1:T}, z_{1:T}\mid h_{1:T})}{q(z_{1:T}\mid b_{1:T}, h_{1:T})}\,dz_{1:T}$$

$$\ge \mathbb{E}_{q(z_{1:T}\mid b_{1:T}, h_{1:T})}\!\left[\log \frac{p(b_{1:T}, z_{1:T}\mid h_{1:T})}{q(z_{1:T}\mid b_{1:T}, h_{1:T})}\right] \equiv \mathcal{L}$$

$$\mathcal{L} = \mathbb{E}_{q(z_{1:T}\mid b_{1:T}, h_{1:T})}\!\left[\log p(b_{1:T}, z_{1:T}\mid h_{1:T}) - \log q(z_{1:T}\mid b_{1:T}, h_{1:T})\right]$$

### 1.1 The joint (generative)

$$p(b_{1:T}, z_{1:T}\mid h_{1:T}) = p(z_1\mid h_{1:T})\,p(b_1\mid z_1)\prod_{t=2}^{T} p(z_t\mid z_{t-1}, h_{1:T})\,p(b_t\mid z_t)$$

$$\log p(b_{1:T}, z_{1:T}\mid h_{1:T}) = \log p(z_1\mid h_{1:T}) + \sum_{t=1}^{T}\log p(b_t\mid z_t) + \sum_{t=2}^{T}\log p(z_t\mid z_{t-1}, h_{1:T})$$

The emission $p(b_t\mid z_t)$ carries **no** $h$: the decoder reads only $z_t$. That absence is
deliberate (so reconstruction cannot bypass the latents), not a dropped subscript.

### 1.2 The posterior (smoothing)

Autoregressive in $z$, but each factor conditions on the whole $(b_{1:T}, h_{1:T})$ because the
context GRU is **bidirectional** — so this is a *smoothing* family, not a filtering one.

$$q(z_{1:T}\mid b_{1:T}, h_{1:T}) = q(z_1\mid b_{1:T}, h_{1:T})\prod_{t=2}^{T} q(z_t\mid z_{t-1}, b_{1:T}, h_{1:T})$$

$$\log q(z_{1:T}\mid b_{1:T}, h_{1:T}) = \log q(z_1\mid b_{1:T}, h_{1:T}) + \sum_{t=2}^{T}\log q(z_t\mid z_{t-1}, b_{1:T}, h_{1:T})$$

### 1.3 Combine

$$\begin{aligned}
\mathcal{L} = \;& \sum_{t=1}^{T}\mathbb{E}_{q(z_t\mid b_{1:T}, h_{1:T})}\!\left[\log p(b_t\mid z_t)\right] \\
&- D_{\mathrm{KL}}\!\left(q(z_1\mid b_{1:T}, h_{1:T})\,\middle\|\,p(z_1\mid h_{1:T})\right) \\
&- \sum_{t=2}^{T}\mathbb{E}_{q(z_{t-1}\mid b_{1:T}, h_{1:T})}\!\left[D_{\mathrm{KL}}\!\left(q(z_t\mid z_{t-1}, b_{1:T}, h_{1:T})\,\middle\|\,p(z_t\mid z_{t-1}, h_{1:T})\right)\right]
\end{aligned}$$

Note the asymmetry inside the transition KL: $q$ carries $b_{1:T}$, the prior $p$ does not. That gap
*is* the KL tax — it penalizes the posterior for using the answer key $b_{1:T}$ to deviate from what
the audio-only prior would predict.

### 1.4 Factorize the per-frame KL

Both densities factor over $z_t = (\dot\phi_t, \phi_t, m_t)$:

$$p(z_t\mid z_{t-1}, h_{1:T}) = p(\dot\phi_t\mid \dot\phi_{t-1}, h_{1:T})\,p(\phi_t\mid \phi_{t-1}, \dot\phi_{t-1}, h_{1:T})\,p(m_t\mid m_{t-1}, \phi_t, \phi_{t-1}, h_{1:T})$$

Below, every $q$-factor conditions on $(z_{t-1}, b_{1:T}, h_{1:T})$; written $q(\cdot)$ to keep the
line readable.

$$\begin{aligned}
D_{\mathrm{KL}}\!\left(q(z_t\mid z_{t-1}, b_{1:T}, h_{1:T})\,\middle\|\,p(z_t\mid z_{t-1}, h_{1:T})\right) = \;& D_{\mathrm{KL}}\!\left(q(\dot\phi_t)\,\middle\|\,p(\dot\phi_t\mid \dot\phi_{t-1}, h_{1:T})\right) \\
&+ D_{\mathrm{KL}}\!\left(q(\phi_t)\,\middle\|\,p(\phi_t\mid \phi_{t-1}, \dot\phi_{t-1}, h_{1:T})\right) \\
&+ \mathbb{E}_{q(\phi_t)}\!\left[D_{\mathrm{KL}}\!\left(q(m_t)\,\middle\|\,p(m_t\mid m_{t-1}, \phi_t, \phi_{t-1}, h_{1:T})\right)\right]
\end{aligned}$$

The meter term keeps the outer $\mathbb{E}_{q(\phi_t)}$ because its prior conditions on the
**current-frame** phase $\phi_t$, which is random under $q$; the tempo and phase terms collapse to
plain marginal KLs, but the meter one does not. In the single-sample rollout this expectation is
estimated at the sampled $\phi_t$ — which is exactly what the code does (`meter KL evaluated AFTER
sampling phi_t`).
