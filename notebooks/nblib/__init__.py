"""Supporting code for the VBPM from-first-principles notebook.

The notebook keeps the *derivation and model* inline (the ELBO, the closed-form KLs, the von Mises
reparameterization, the model class, the objective, the training harness). Everything that is
plumbing rather than MVP -- data loading, plotting, deployment scoring, and the against-the-data
diagnostic graphs -- lives here as small importable modules, so the notebook reads as a clean
narrative while the code stays one folder over, browsable in full.

Modules: setup (constants/units/palette/seeding), data (Song + loaders), evaluation (geometric
read-out + F-measure), plotting (all figures), diagnostics (annotation-fit graphs + the ELBO probe).
"""
