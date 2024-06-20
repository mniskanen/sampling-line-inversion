# -*- coding: utf-8 -*-

import numpy as np

from numba import njit
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from scipy.interpolate import RBFInterpolator


BOLTZMANN_CONSTANT = 1.380649e-23  # Boltzmann constant
PARTICLE_DENSITY = 2e3  # Density of a spherical soot particle (kg m^-3)


class SamplingLine():
    ''' A model for the sampling line and the processes an aerosol sample
    flowing through it experiences. '''
    
    def __init__(self,
                 particle_mobility_diameter,  # meters
                 sampling_line_length,  # meters
                 sampling_line_diameter,  # meters
                 flow_velocity  # meters / second
                 ):
        
        assert particle_mobility_diameter.ndim == 1, 'Particle mobility diameter should be ' \
            f'a one-dimensional vector, got ndim=={particle_mobility_diameter.ndim}'
        
        # Mobility radius, i.e. what is (usually) measured. This is related to the measuring
        # instrument and not changed within this class.
        self.r_m = particle_mobility_diameter[:, np.newaxis] / 2
        
        
        # Read in basic properties of the sampling line (with rudimentary error checking)
        
        if sampling_line_length >= 0:
            # Allow zero length here to 'turn off' the sampling line
            self.sampling_line_length = sampling_line_length
        else:
            raise ValueError('Sampling line length cannot be negative')
        
        if sampling_line_diameter > 0 and sampling_line_diameter < 1:
            self.sampling_line_diameter = sampling_line_diameter
        else:
            raise ValueError('Are you sure the diameter of your sampling line is correct?')
        
        if flow_velocity > 0:
            self.flow_velocity = flow_velocity
        else:
            raise ValueError('Flow velocity has to be positive')
        
        
        # Derived sampling line values
        self.residence_time = self.sampling_line_length / self.flow_velocity
        self.volume_flow_rate = self.flow_velocity * np.pi / 4 * self.sampling_line_diameter**2
        
        # Optional interpolator to compute r_v faster
        self.use_rm_rv_map = False
        self.rm_rv_interpolator = None
        self.scale_r1 = 1e8  # A scaling factor for r_1 when using the interpolator
        
        # Default values for the aerosol model
        self.temperature = 250 + 273  # Exhaust gas temperature (K)
        self.D_f = 3.0  # Fractal dimension
        self.r_1 = 27e-9 / 2  # Primary particle radius (m)
        self.hamaker = 2e-19  # Hamaker constant
        
        # Derived aerosol model values. These depend on the model parameters and thus can change.
        self.r_v = None  # Volume-equivalent radius
        self.r_f = None  # Fractal radius
        self.diffusion_coefficient = None  # (Fractal) diffusion coefficient
        self.diffusion_P = None  # Surviving fraction of particles after diffusion
        self.diffusion_R = None  # Diffusion removal coefficient
        self.K = None  # Coagulation coefficient
        self.xi = None  # Size-splitting coefficient
        
        # To store a previously computed coagulation kernel (for computational reasons in MCMC)
        self.stored_K = None
        self.stored_xiK = None
        
        self.update_aerosol_model()
    
    
    def store_current_parameters(self):
        """ Save the calculated parameters (coagulation coefficient, diffusion, size-splitting,
        ...) to speed up MCMC evaluations. During MCMC, we sometimes propose an update to these
        model parameters, and if the proposal is not accepted, we have to return back to the
        previous ones. Using the stored values, we can avoid (re-doing) the expensive model
        update calculations."""
        
        # Input parameters
        self.stored_temperature = self.temperature
        self.stored_r_1 = self.r_1
        self.stored_D_f = self.D_f
        self.stored_hamaker = self.hamaker
        self.stored_flow_velocity = self.flow_velocity
        self.stored_residence_time = self.residence_time
        self.stored_volume_flow_rate = self.volume_flow_rate
        
        # Computed parameters
        self.stored_r_v = self.r_v
        self.stored_xi = self.xi
        self.stored_r_f = self.r_f
        self.stored_diffusion_coefficient = self.diffusion_coefficient
        self.stored_diffusion_P = self.diffusion_P
        self.stored_diffusion_R = self.diffusion_R
        self.stored_K = self.K
        self.stored_xiK = self.xiK
    
    
    def restore_stored_parameters(self):
        # Input parameters
        self.temperature = self.stored_temperature
        self.r_1 = self.stored_r_1
        self.D_f = self.stored_D_f
        self.hamaker = self.stored_hamaker
        self.flow_velocity = self.stored_flow_velocity
        self.residence_time = self.stored_residence_time
        self.volume_flow_rate = self.stored_volume_flow_rate
        
        # Computed parameters
        self.r_v = self.stored_r_v
        self.xi = self.stored_xi
        self.r_f = self.stored_r_f
        self.diffusion_coefficient = self.stored_diffusion_coefficient
        self.diffusion_P = self.stored_diffusion_P
        self.diffusion_R = self.stored_diffusion_R
        self.K = self.stored_K
        self.xiK = self.stored_xiK
    
    
    def update_aerosol_model(self, temperature=None, fractal_dim=None,
                             primary_particle_diam=None, hamaker_constant=None,
                             flow_velocity=None, compute_vdw=True
                             ):
        
        # Read in possible new values
        if temperature:
            if temperature > 0:
                self.temperature = temperature
            else:
                raise ValueError('Negative temperature')
        
        if primary_particle_diam:
            if primary_particle_diam > 0:
                self.r_1 = primary_particle_diam / 2
            else:
                raise ValueError('Negative primary particle diameter')
        
        if fractal_dim:
            if fractal_dim >= 1.0 and fractal_dim <= 3.0:
                self.D_f = fractal_dim
            else:
                raise ValueError('Illegal fractal dimension value')
        
        if hamaker_constant:
            if hamaker_constant > 0:
                self.hamaker = hamaker_constant
            else:
                raise ValueError('Negative Hamaker constant')
        
        if flow_velocity:
            if flow_velocity > 0:
                self.flow_velocity = flow_velocity
                
                # Derived sampling line values
                self.residence_time = self.sampling_line_length / self.flow_velocity
                self.volume_flow_rate = (self.flow_velocity * np.pi / 4
                                         * self.sampling_line_diameter**2)
            else:
                raise ValueError('Non-positive flow velocity')
        
        
        # Compute parameters for the aerosol model equation
        
        if self.use_rm_rv_map:
            self.r_v = self.compute_rv_interpolate(self.r_1, self.D_f)
        else:
            self.r_v = self.compute_rv_minimize(self.r_m, self.r_1, self.D_f, self.temperature)
        
        self.xi = self.compute_size_splitting_coefficient(2 * self.r_v)
            
        self.r_f = self.compute_fractal_radius()
        
        self.diffusion_coefficient = self.compute_diffusion_coefficient(self.r_m)
        
        # Particle penetration factor P
        self.diffusion_P = self.compute_penetration_due_to_diffusion()
        
        # Removal coefficient R
        self.diffusion_R = self.convert_penetration_to_removal()
        
        self.K = SamplingLine.compute_coagulation_coefficient_agglomerates(
            self.r_m, self.r_f, self.r_v, self.diffusion_coefficient,
            self.temperature, self.hamaker, compute_vdw)
        
        # Precompute for faster forward model evaulation
        self.xiK = np.zeros_like(self.xi)
        for i in range(len(self.r_v)):
            self.xiK[i] = self.xi[i] * self.K
    
    
    def prepare_for_mcmc(self):
        """ Precompute a mapping (using interpolation) from r_m to r_v for a range of r_1 and D_f.
        This takes a while but after it's done it speeds up considerably the evaluation of
        update_aerosol_model(), which we will need to evaluate almost every MCMC iteration
        when estimating SamplingLine parameters. """
        
        self.rm_rv_interpolator = self.generate_rm_to_rv_interpolator()
        self.use_rm_rv_map = True
        print('rm to rv interpolator generated')
    
    
    def store_coagulation_coefficient(self):
        if self.K is not None:
            self.stored_K = self.K.copy()
            self.stored_xiK = self.xiK.copy()
        else:
            raise ValueError('Cannot store an empty coagulation kernel.')
    
    
    def restore_coagulation_coefficient(self):
        if self.stored_K is not None:
            self.K = self.stored_K
            self.xiK = self.stored_xiK
            
            # This should only be usable once after storing. Clear values to ensure that.
            self.stored_K = None
            self.stored_xiK = None
            
        else:
            raise ValueError('Cannot restore an empty coagulation kernel.')
    
    
    def compute_penetration_due_to_diffusion(self):
        """
        Compute the fraction P of particles that penetrate a cylindrical tube whose walls the
        particles diffuse on.
        Penetration P is computed from Hinds - Aerosol technology (1999), pg. 163.
        """
        
        # Deposition parameter (dimensionless)
        mu = self.diffusion_coefficient * self.sampling_line_length / self.volume_flow_rate
        mu = np.squeeze(mu)
        
        P = np.zeros(len(self.diffusion_coefficient))
        
        # mu < 0.009
        idx_s = np.where(mu < 0.009)
        P[idx_s] = 1 - 5.50 * mu[idx_s]**(2/3) + 3.77 * mu[idx_s]
        
        # mu >= 0.009
        idx_l = np.where(mu >= 0.009)
        P[idx_l] = 0.819 * np.exp(-11.5 * mu[idx_l]) + 0.0975 * np.exp(-70.1 * mu[idx_l])
        
        return P
    
    
    def convert_penetration_to_removal(self):
        """ Convert penetration P (fraction of particles that survie through the sampling line)
        into R, the rate at which particles are removed so that a fraction P remains after time
        residence_time. """
        
        if self.residence_time > 0:
            return -np.log(self.diffusion_P) / self.residence_time
        else:
            return np.zeros_like(self.diffusion_P)
    
    
    @staticmethod
    def compute_rv_minimize(r_m, r_1, D_f, temperature):
        """ Compute r_v from given r_m, r_1, D_f, and temperature via minimization. """
        
        # A function to compute the difference between a desired r_m and r_m computed from r_v
        def f(r_v, r_m, r_1, D_f, temperature):
            r_m_model = SamplingLine.spherical_to_mobility_radius(r_v, r_1, D_f, temperature)
            return np.abs(r_m - r_m_model.flatten())
        
        # Options for minimizer
        options = {'xatol' : 1e-11}
        max_bound = np.max(r_m)  # r_v can't be larger than r_m
        min_bound = 0.1 * np.min(r_m)
        
        r_v =  np.zeros_like(r_m)
        for i in range(len(r_m)):
            res = minimize_scalar(f,
                                  bounds=(min_bound, max_bound),
                                  args=(r_m[i], r_1, D_f, temperature),
                                  method='bounded',
                                  options=options
                                  )
            r_v[i] = res.x
        
        return r_v
    
    
    def compute_rv_interpolate(self, r_1, D_f):
        """ Use radial basis function interpolation to convert the given mobility radius to
        the spherical (volume-equivalent) radius.
        """
        
        r_v =  np.zeros_like(self.r_m)
        for i in range(len(self.r_m)):
            r_v[i] = self.rm_rv_interpolator[i](np.array([[r_1 * self.scale_r1, D_f]]))
        
        return r_v
    
    
    def generate_rm_to_rv_interpolator(self):
        """ Generate a radial basis function interpolator to map (in a computationally efficient
        way) a given mobility radius r_m, primary particle radius r_1, and fractal dimension D_f,
        to the volume-equivalent radius r_v. Temperature T does also affect the conversion a bit
        (via mean free path), but its effect over the range we're interested in is insignificant
        compared to those of r_1 and D_f. Therefore, we can set T to a representative value and
        keep it constant to furhter speed up the calculations.
        """
        
        N_r_m = self.r_m.shape[0]  # Number of particle mobility sizes
        N_r_1 = 10  # Number of primary particle sizes
        N_D_f = 15  # Number of fractal dimensions
        
        # Specify the ranges of r_1 and D_f over which we calculate the map
        r_1 = np.linspace(5e-9, 20e-9, N_r_1)
        D_f = np.linspace(1, 3, N_D_f)
        T = 273 + 250  # Representative (average over the considered range) temperature
        
        # Evaluation points for RBF
        # Scale r_1 to be of the same order as D_f for numerical accuracy
        pts = np.meshgrid(r_1 * self.scale_r1, D_f, indexing='ij')
        pts = np.c_[pts[0].ravel(), pts[1].ravel()]
        RBF_interp = []
        
        # For each r_m[i]
        for i in range(N_r_m):
            # Find the r_v that maps to r_m[i] for a given r_1[j] and D_f[k]
            r_v = np.zeros((N_r_1, N_D_f))
            for j in range(N_r_1):
                for k in range(N_D_f):
                    r_v[j, k] = self.compute_rv_minimize(self.r_m[i], r_1[j], D_f[k], T)
            
            # Construct the RBF interpolator (separately for each r_m[i])
            RBF_interp.append(
                RBFInterpolator(pts, r_v.ravel(), kernel='thin_plate_spline', degree=4)
                )
        
        return RBF_interp
    
    
    @staticmethod
    def spherical_to_mobility_radius(r_v, r_1, D_f, T):
        """ Compute the mobility radius r_m for fractal-like agglomerates.
        r_v = volume-equivalent (spherical) radius
        r_1 = primary particle radius
        D_f = fractal dimension
        T = temperature (K)
        
        Static method because we have to be able to call this with many different values.
        """
        
        if np.isscalar(r_v):
            r_v = np.array([r_v])
        r_v = r_v[:, np.newaxis]  # Convert to a row vector (for broadcasting later)
        assert r_v.shape == (len(r_v), 1)
        
        # Number of primary particles in an agglomerate of given size (modelling the aggregate as
        # a sphere)
        N = (r_v / r_1)**3
        r_f = r_1 * N**(1 / D_f)  # Fractal radius
        r_f = np.max((r_v, r_f), axis=0)  # When radius is smaller than radius of the primary
                                          # particle, use the spherical radius
        
        # Preallocate projected area diameter
        r_A = r_v.copy()
        
        # Only do these calculations if at least one of the modelled volume-equivalent particles is
        # larger than the the primary particle
        if r_1 < r_v[-1]:
            alpha = 0.67
            start = np.where(r_1 < r_v)[0][0]
            r_A[start:] = r_1 * np.sqrt(
                np.max((N[start:]**(2 / 3),
                        np.min((1 + alpha * (N[start:] - 1),
                                D_f / 3 * N[start:]**(2 / D_f)
                                ), axis=0)
                        ), axis=0)
                )
            
            r_mc = np.max((r_f / (np.log(2 * r_f / r_1) + 1),
                           r_f * ((D_f - 1) / 2)**0.7,
                           r_A
                           ), axis=0)
            
            mean_free_path = SamplingLine.mean_free_path_air(T)
            
            # Particle mobility radius (with transition regime correction factor from R&F).
            # r_m is implicitly defined so we have to compute it iteratively. Fixed-point
            # iteration seems to oscillate around the truth (converged after ~500 iterations),
            # and for some reason taking a geometric mean of the first two iterations seems to
            # be close to the truth.
            r_m0 = r_mc * SamplingLine.cunningham(mean_free_path / r_mc) \
                        / SamplingLine.cunningham(mean_free_path * r_mc / r_A**2)
            r_m1 = r_mc * SamplingLine.cunningham(mean_free_path / r_m0) \
                        / SamplingLine.cunningham(mean_free_path * r_mc / r_A**2)
            
            # Geom. mean of two first iterations
            r_m = 10**(0.5 * (np.log10(r_m0) + np.log10(r_m1)))
            
        else:
            r_m = r_v
        
        return r_m
    
    
    def compute_fractal_radius(self):
        """ Compute the fractal radius r_f from spherical radius, primary particle radius, and
        fractal dimension. """
        
        # Number of primary particles
        N = (self.r_v / self.r_1)**3
        r_f = self.r_1 * N**(1 / self.D_f)
        
        # When r_f would be smaller than radius of the primary particle, use the spherical radius
        r_f = np.max((self.r_v, r_f), axis=0)
        
        return r_f
    
    
    def compute_diffusion_coefficient(self, r):
        """ Compute the Brownian diffusion coefficient for particle radius r (given as input
        here so that we can compute this for different radii).
        """
        
        Kn = SamplingLine.mean_free_path_air(self.temperature) / r
        mu = SamplingLine.dynamic_viscosity(self.temperature)
        Cc = SamplingLine.cunningham(Kn)
        
        return BOLTZMANN_CONSTANT * self.temperature / (6 * np.pi * r * mu) * Cc
    
    
    @staticmethod
    def compute_coagulation_coefficient_agglomerates(r_m, r_f, r_v, D, T, A, compute_vdw):
        """ Compute the coagulation coefficient in the transition regime after Rogak & Flagan
        (1992), with notation from Jacobson & Seinfeld (2004) and using the Fuchs model instead of
        the Dahneke model used in R&K.
        Additionally, can compute the van der Waals/viscous correction factor V, as detailed in
        Jacobson & Seinfeld (2004).
        Focus is on fast computation because we may want to sample this function a lot of times.
        In this function we use radius as particle size, to be consistent in notation with J&S.
        
        Input:
            r_m = mobility radius
            r_f = fractal radius
            r_v = volume-equivalent radius (spherical)
            D = diffusion coefficient
            T = temperature in Kelvins
            A = Hamaker constant
            compute_vdw = (bool), if True, compute van der Waals and viscous effects
        
        Returns:
            K = matrix of coagulation coefficients in cm^3 s^-1
        
        """
        
        # Assumption (from J&S and R&F Figure 10): collision radius == fractal (outer) radius
        r_c = r_f
        
        # Particle volume
        v_p = 4 * np.pi / 3 * r_v**3
        m_particle = PARTICLE_DENSITY * v_p  # Particle mass
        
        # Mean thermal speed
        c = np.sqrt(8 * BOLTZMANN_CONSTANT * T / (np.pi * m_particle))
        
        # Effective mean free path
        lmbd_m = 8 * D / (np.pi * c)
        
        delta_m = ((2 * r_m + lmbd_m)**3 - (4 * r_m**2 + lmbd_m**2)**(3/2)) \
                  / (6 * r_m * lmbd_m) - 2 * r_m
        
        # Broadcasting
        r_c12 = r_c + r_c.T
        D12 = D + D.T
        
        helper_term = 4 * D12 / (np.sqrt(c**2 + c.T**2) * r_c12)
        
        K = 4 * np.pi * r_c12 * D12 / (
            r_c12 / (r_c12 + np.sqrt(delta_m**2 + delta_m.T**2)) + helper_term
            )
        
        K *= 1e6  # Transform from m^3 s^-1 to cm^3 s^-1
        
        if compute_vdw:
            # van der Waals + viscous effects
            V = SamplingLine._compute_vdW_viscous_correction(r_c, A, T, helper_term)
            K *= V
        
        return K
    
    
    @staticmethod
    @njit(cache=True)
    def _compute_vdW_viscous_correction(r_c, A, T, helper_term):
        # van der Waals + viscous effects
        # Here the radius used is taken to be the collision (fractal) radius.
        
        # Discretisation for the integral
        n_points = 500
        r_max = 5e-5
        
        # print(f'A / kT = {A / (BOLTZMANN_CONSTANT * T) : .0f}')
        
        # Compute in two for loops (this seems to be faster than numpy broadcasting when
        # number of integration points is large enough, and especially when using numba).
        # W_k & W_c are symmetric matrices, so let's just compute the upper halves
        # (incl. diagonals) first
        W_k = np.zeros((r_c.shape[0], r_c.shape[0]), dtype=np.float64)
        W_c = np.zeros_like(W_k)
        n = r_c.shape[0]
        for i in range(n):
            for j in range(i, n):
                r = np.logspace(np.log10((1 + 1e-4) * (r_c[i] + r_c[j]))[0],
                                np.log10(r_max),
                                n_points
                                )
                r2 = r**2
                
                a = 2 * r_c[i] * r_c[j]
                p = (r_c[i] + r_c[j])**2
                m = (r_c[i] - r_c[j])**2
                
                E = -A / 6 * (
                    a / (r2 - p) + a / (r2 - m) + np.log((r2 - p) / (r2 - m))
                    )
                
                termm = 1 / (r2**2 - r2 * (m + p) + m * p)
                dE = -2 * a * r / (r2 - p)**2 - 2 * a * r / (r2 - m)**2 \
                    + 2 * r * (p - m) * termm
                dE = -A / 6 * dE
                
                d2E = 8 * a * r2 / (r2 - p)**3 - 2 * a / (r2 - p)**2 \
                    + 8 * a * r2 / (r2 - m)**3 - 2 * a / (r2 - m)**2 \
                    + 2 * (p - m) * termm * (1 - (4 * r2**2 - 2 * r2 * (m + p)) * termm)
                d2E = -A / 6 * d2E
                
                rterm = r_c[i] * r_c[j] / ((r_c[i] + r_c[j]) * (r - r_c[i] - r_c[j]))
                DperD = 1 + 2.6 * r_c[i] * r_c[j] / (r_c[i] + r_c[j])**2 * np.sqrt(rterm) + rterm
                
                # Trapezoidal integration
                delta_r = np.diff(r)
                integrand_wk = (dE + r * d2E) * np.exp(
                    -1 / (BOLTZMANN_CONSTANT * T) * (r / 2 * dE + E)
                    ) * r2
                W_k[i, j] = (-1 / (2 * (r_c[i] + r_c[j])**2 * BOLTZMANN_CONSTANT * T) \
                    * 0.5 * np.sum((integrand_wk[1:] + integrand_wk[:-1]) * delta_r))[0]
                
                integrand_wc = DperD * np.exp(E / (BOLTZMANN_CONSTANT * T)) / r2
                W_c[i, j] =  1 / ((r_c[i] + r_c[j]) * 0.5 * np.sum(
                    (integrand_wc[1:] + integrand_wc[:-1]) * delta_r))[0]
        
        # Fill in the lower half and remove doubling from the diagonal
        W_k = W_k + W_k.T
        W_k = W_k - 0.5 * np.diag(np.diag(W_k))
        W_c = W_c + W_c.T
        W_c = W_c - 0.5 * np.diag(np.diag(W_c))
        
        # Van der Waals/viscous correction factor
        V = W_c * (1 + helper_term) / (1 + W_c / W_k * helper_term)
        
        return V
    
    
    def compute_coagulation_coefficient_spherical(self, d_v):
        """ Compute the coagulation coefficient in the continuum regime after Seinfeld & Pandis
        Table 13.1.
        
        Input: d_v = particle diameter in meters
        Returns: K = matrix of coagulation coefficients in cm^3 s^-1
        
        """
        
        n = len(d_v)
        
        v_p = np.pi / 6 * d_v**3  # Particle volumes assuming spherical particles
        m = PARTICLE_DENSITY * v_p  # Particle mass
        
        # Brownian diffusion coefficient (use spherical radius here)
        D = self.compute_diffusion_coefficient(d_v / 2)
        
        # Mean thermal speed
        c = np.sqrt(8 * BOLTZMANN_CONSTANT * self.temperature / (np.pi * m))
        
        l = 8 * D / (np.pi * c)
        g = 1 / (3 * d_v * l) * ((d_v + l)**3 - (d_v**2 + l**2)**(3 / 2)) - d_v
        # There's an error in the formula of g in S&P 3rd edition and the factor should be
        # 1 instead of sqrt(2)
        
        K = np.zeros((n, n))
        for i in range(n):
            K[i, :] = np.squeeze(2 * np.pi * (D[i] + D) * (d_v[i] + d_v) / (
                (d_v[i] + d_v) / (d_v[i] + d_v + 2 * (g[i]**2 + g**2)**(1 / 2))
                + 8 * (D[i] + D) / ((c[i]**2 + c**2)**(1 / 2) * (d_v[i] + d_v))
                ))
        
        K *= 1e6  # Transform from m^3 s^-1 to cm^3 s^-1
        
        return K
    
    
    def compute_size_splitting_coefficient(self, d_v):
        """ Compute xi, a N^3 size-splitting matrix that describes the volume fraction of a
        coagulated pair i,j partitioned into bin k.
        
        Input : particle (spherical, volume-equivalent) diameter
        """
        
        # Particle volume
        v_p = np.pi / 6 * d_v**3
        n = len(v_p)
        
        # Every xi[k] is symmetric, so to speed up we compute just the upper diagonal here
        xi = np.zeros((n, n, n))
        for i in range(n):
            for j in range(i, n):
                v_new = v_p[i] + v_p[j]
                
                # If the new volume is larger than the maximum considered one, just dump the new
                # particle to the largest volume bin (to conserve volume). Another option would be
                # to compute the (possible) contribution of the new particle to the largest volume
                # bin and let the rest disappear from the system.
                if v_new > v_p[-1]:
                    xi[-1, i, j] = 1
                    
                else:
                    # Finds the first index minus one whose volume is greater than new_vol
                    k = np.argmax(v_p > v_new) - 1
                    
                    alpha = (v_p[k + 1] - v_new) / (v_p[k + 1] - v_p[k])
                    beta = 1 - alpha
                    
                    xi[k, i, j] = alpha
                    xi[k + 1, i, j] = beta
        
        # Fill in the lower half and remove doubling from the diagonal
        for k in range(n):
            xi[k] = xi[k] + xi[k].T - np.diag(np.diag(xi[k]))
        
        return xi
    
    
    @staticmethod
    def coagulation_diffusion_equation(t, N_0, xiK, K, diffusion_R):
        """ Coagulation and diffusion in a form that can be given to solve_ivp. """
        
        return (0.5 * N_0 @ xiK - np.diag(N_0) @ K - np.diag(diffusion_R)) @ N_0
    
    
    def run(self, N_0, return_Jacobian=False):
        """ Solve the coagulation and diffusion equations from time 0 to residence_time. """
        
        sol = solve_ivp(SamplingLine.coagulation_diffusion_equation,
                        [0, self.residence_time],
                        N_0,
                        method='RK45',
                        args=(self.xiK, self.K, self.diffusion_R),
                        rtol=1e-10, atol=1e-5
                        )
        
        if return_Jacobian:
            J = SamplingLine.compute_Jacobian(sol.t, sol.y, self.xiK, self.K, self.diffusion_R)
            return sol.y[:, -1], J
        
        return sol.y[:, -1]
    
    
    @staticmethod
    @njit(cache=True)
    def compute_Jacobian(time_points, N, xiK, K, diffusion_R):
        
        len_N = K.shape[0]
        I = np.eye(len_N)
        J = np.eye(len_N)
        diffusion_R = np.diag(diffusion_R)
        dt = np.diff(time_points)
        n_iter = time_points.shape[0]
        broadcast = np.zeros((len_N, len_N))
        for t in range(n_iter-1, 0, -1):
            for i in range(len_N):
                broadcast[i] = N[:, t] @ xiK[i]
            
            J = J @ (
                I + dt[t-1] * (
                    broadcast - (
                        np.diag(N[:, t]) @ K + np.diag(K @ N[:, t])
                        + diffusion_R
                        )
                    )
                )
        
        return J
    
    
    @staticmethod
    @njit(cache=True)
    def dynamic_viscosity(temperature):
        """ Dynamic viscosity of air given by the Sutherland formula. """
        return 1.458e-6 * temperature**1.5 / (temperature + 110.4)
    
    
    @staticmethod
    @njit(cache=True)
    def mean_free_path_air(temperature):
        """ Calculate the mean free path of an air molecule for a given temperature. Assume that
        the air molecules consist of nitrogen only.
        """
        # Air pressure (Pa)
        p = 101e3
        
        # Diameter of a nitrogen molecule (adjusted so that we get a mean free path of 68 nm for
        # temperature 20 C)
        d = 3.64e-10
        
        return BOLTZMANN_CONSTANT * temperature / (np.sqrt(2) * np.pi * d**2 * p)
    
    
    @staticmethod
    @njit(cache=True)
    def cunningham(Kn):
        """Compute the Cunningham slip correction factor for given Knudsen numbers Kn."""
        return 1 + Kn * (1.257 + 0.4 * np.exp(-1.1 / Kn))
