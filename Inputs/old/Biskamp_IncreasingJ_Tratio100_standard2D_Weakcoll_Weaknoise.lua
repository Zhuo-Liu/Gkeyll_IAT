-- Gkyl ------------------------------------------------------------------------
--Notes
-- 1. Biskamp had a total run time of t = 2000 omega_pe^-1.  His plots show that the fluctuation energy increases until t = 800-1000, and then there is an ion heating phase, which is also a runaway production phase.
--      1.1. Biskamp used a mass ratio of 400 for the plots shown.  The meaningful time is omega_pi, so t = 2000 (Sqrt[m_e/m_i]Biskamp) omega_pe^-1 = 2000*Sqrt(25/400) omega_pe^-1 = 2000/4 omega_pe^-1 = 500 omega_pe^-1. 
--      1.2. We can run the sim for 1/2 the time, make plots, then run for the remaining 1/2 time.
-- 2. The ion-ion collision freequency does not have T dependence (so it is initially too small by a factor of 1000)
--      2.2 We want this simulation to be as similar as possible to the previous simulation.
--      2.3. The relevant speeds for the ions is the sound speed.  It is non_Maxwellian features that form here, in the high energy tail.  This probably does not matter for what we are interested in: we can learn from it without publishing i
-- 3. The time step is 0.0028125 omega_pe^-1.
--      3.1 This is good to know for choosing scale of stochastic noise
-- 4. Biskamp had a spatial grid of 128x128.  We have a grid of 32*3x16*3 = 96*45
-- 5. We have an electron v space grid of 48*3X48*3 = 144x144.  This is sort of equivalent to 20736 particles in a region Delta x * Delta y (where this is the effective Delta x_alpha).  In the region of velocity space where most electrons are, there are (2 sigma)^2/(6 sigma)^2 particles (i.e. 2073 particles).  To convert this to an effective total number of particles in the box, we multiply by the number of effective area cells = 32*3 * 16*3 = 4603.  Finally, we find a total number of particles = 95,551,488 and a total effective number of particles = 9,555,149.  Biskamp has 2*10^5 particles, and nearly all of them are in the circle of radius 2sigma. So we resolve velocity space by a factor of 10 in each dimension in the region where most of the particles are.  Biskamp has more spatial resolution than us, but less overall phase space resolution and quite a bit less velocity space reolution.
--      5.1 The resonant ions are fairly deep in the tail of the initial ion distribution.  Here, a PIC approximation would be quite inaccurate (because it would take a large region to have 100 superparticles)
--           5.1.1. So how can Biskamp claim to see saturation via ion trapping???
-- 6. Biskamp has unavoidable noise due to having a rather low number of particles in the Debye sphere.  The scale of this noise is delta E^2 = T_e / lambda_De ^3.  
--      6.1. The effective nu_e/i is nu_e/i = omega_pe * LogLambda/N_D = omega_pe * 10 /   
--      6.2. Our collision frequency is nu_e/i = 10^-2
--           6.2.1. This should produce a DC electric field with magnitude e_e E =  m_e nu_e/i u_drift ==> E = 0.02*0.01 = 0.0002 in code units.
--           6.2.2. Biskamps peak DC electric field is 0.03.  When you do the conversion, this is nu_eff/omega_pe = 0.03.  So he obtains a DC electric field that is 3 times the value we expect to obtain from the Lenard-Bernstein operator.
--           6.2.3. The box-averaged distribution functions are quite smooth in velocity space, but the local in space distributions are not far from the grid scale.  This makes sense: the ion-acoustic eigenfunctions have poles at the resonant velocity in the collisionless limit.  (More reason to distrust PIC)
-- 7. Biskamp has a box length = L = 128 * 0.4 * lambda_De = 51.2 lambda_De.  Our box size is d_e = 50 lambda_de.  Basically identical.
-- 8. Our smallest spatial scale is Delta x = 50 Lambda_De / (32*3) = 0.52 Lambda_De.  The most unstable wave will have wavelength that is very very close to lambda_De.
--     8.1 For a temperature ratio of 100, the most unstable wavevector satisfies k_Crit lambda_D = 0.2.  However, we know that the ions will heat up by a factor of 5-10, which will increase the most unstable wavelength.  So we are at the margins perhaps.
-- 9. The rate at which the current would increase if there was inflows corresponding to v_in = 0.1 v_A and v_inf/c = Omega_i/omega_pi = 0.1 is 
--                             \omega_pe \tau_rec = 500    ==> alpha = 1/500 = 0.002
--    I get this via tau_rec = 10 d_i / v_inf = 10 omega_pi^-1 /(v_inf/c) = 10 (omega_pi/omega_pe)^-1 omega_pe^-1 / 0.1 = 100 * sqrt[m_i/m_e] omega_pe^-1 = 500 omega_pe^-1.
local Plasma = require("App.PlasmaOnCartGrid").VlasovMaxwell



massRatio = 25.0
vTe = 0.02                                                 --This sets lambda_e/d_e
tempRatio = 100.0
alpha = 0.002               --this is the rate at which the external current is increased (in units of omega_pe^-1)
                          --so the current increases by e n vTe every omega_pe T = 1/0.002 = 500 ---> omega_pi T = 100 ---> runtime will be 4.9*T = 490



cSound = vTe/math.sqrt(massRatio)                          --RLW: This line is only needed for the line below
uDrift = 0.2*cSound                                        --RLW: 
uFinal = 1.0*vTe
runTime =2.0*(uFinal - uDrift)/(vTe*alpha)                  --This is equivalent to Biskamps run time with his mass ratio

-- Electron parameters
vDriftElc = {uDrift, 0.0}                                  --RLW
vtElc     = vTe

-- Ion parameters.
vDriftIon = {0.0, 0.0}
vtIon     = vtElc/ (math.sqrt(massRatio * tempRatio))      -- Modified from 0.001 (use Te = 50 Ti).

-- Collision frequencies.
nuee = 0.0001
nuei = nuee              --RLW 
nuii = nuee/math.sqrt(massRatio)  --should have been nuee/math.sqrt(massRatio)*math.pow(tempRatio,1.5)
nuie = nuee/massRatio

-- Initial Electric field for force balance (accounting for initial increasing current giving an inertia alpha*Drift) 
Ezero = -(nuei*uDrift + alpha*vTe)

perturbation = 1.0e-4   --Scale of electric field perturbation at t = 0 (Note that this isn't needed here)
noise = 1.0e-5          --scale of the fluctuating antenna current

Lz = 1.0                --RLW: This sets lambda_max/lambda_e = 1.0/0.02 = 50
Ly = 0.5                --RLW:
                        -- The most unstable wave is going to satisfy kmax lambda_De = 0.2-1
                        -- This implies lambda_max = 2*pi/(0.1) - 2*pi/1 = 6-30 lambda_De
                        -- In code units, this is 0.12 - 0.6
                        -- I really think that our box size is pretty good, but if anything, we might want to make it bigger.  Perhaps up to 2?

pOrder       = 2        -- Polynomial order.

-- Configuration space grid.
xMin         = {0.0, 0.0}
xMax         = {Lz, Ly}  --RLW
nx           = {32, 16}
lx           = {xMax[1]-xMin[1],xMax[2]-xMin[2]}

-- Velocity space grid.
nvElc        = {48, 48}
vMinElc      = {-6.0*vtElc,-6.0*vtElc}
vMaxElc      = { 6.0*vtElc, 6.0*vtElc}
nvIon        = {32, 32}
vMinIon      = {-32.0*vtIon,-32.0*vtIon}   --#note that this is {-3 cSound, 3 cSound} We need to capture the resonant ions...(There is no way Biskamp can capture this accurately)
vMaxIon      = { 32.0*vtIon, 32.0*vtIon}

randSeed = 42 -- Random seed for random phases in randkick and Drandkick
              -- seed needs to be the same for both so that they get the
              -- same phases for each k.

local function maxwellian2v(v, vDrift, vt)
   return 1/(2*math.pi*vt^2)*math.exp(-((v[1]-vDrift[1])^2+(v[2]-vDrift[2])^2)/(2*vt^2))
end


-----
--
--

local function FluctuatingCurrentDrive(x_table, t, lx_table, ncells_table, p)
   -- x_table = {x, y} are independent variables.
   -- t is the time
   -- lx_table = {xmax-xmin, ymax - ymin}
   -- ncells_table = {ncellsx, ncellsy}
   -- p is polynomial order.
   ------------------------
   -- returns fluctuating random current density {\delta J_x,\delta J_y} at {x, y, t}
   local Nx = ncells_table[1]*(p+1) -- Number of spatial degrees of freedom along x
   local Ny = ncells_table[2]*(p+1) -- Number of spatial degrees of freedom along y
   local Lx, Ly = lx_table[1], lx_table[2]
   local x, y = x_table[1], x_table[2]
   local modJ = 0.0             -- Initialize |J|.
   local Jx, Jy = 0.0, 0.0
   math.randomseed(math.floor(100000*t)) --I want all xs and ys to see the same random phase at a given time step
   for nx = 1, math.floor(Nx/4) do
      for ny = 1, math.floor(Ny/4) do
        modJ = math.random()*math.sin(2*math.pi*(nx*x/Lx + math.random()))*math.sin(2*math.pi*(ny*y/Ly + math.random()))
	--magnitude of J for these mode numbers
        Jx = Jx + mo dJ * nx/ Lx / math.sqrt(nx/ Lx * nx/ Lx + ny/ Ly * ny/ Ly)  --add the component for this particular (nx,ny)
        Jy = Jy + modJ * ny/ Ly / math.sqrt(nx/ Lx * nx/ Lx + ny/ Ly * ny/ Ly) --Note that (delta J)_k should be parallel to k
      end
   end
   return {Jx, Jy}
end

--
--
----
plasmaApp = Plasma.App {
   logToFile = true,

   tEnd        = runTime,          -- End time. RLW: I changed this, but didn't change nFrame below.  So we might get more frames.
   nFrame      = 2000,             -- Number of output frames.  This is 0.5 frames in unit time --> 3 frames every Langmuir period. 
   nDistFuncFrame = 200,           -- Number of distribution function output frames 
   lower       = xMin,             -- Configuration space lower left.
   upper       = xMax,             -- Configuration space upper right.
   cells       = nx,               -- Configuration space cells.
   basis       = "serendipity",    -- One of "serendipity" or "maximal-order".
   polyOrder   = pOrder,           -- Polynomial order.
   timeStepper = "rk3",            -- one of "rk2" or "rk3".
   cflFrac     = 0.9,

   -- Decomposition for configuration space.
   decompCuts = {8,8},    -- Cuts in each configuration direction.
   useShared  = false,    -- If to use shared memory.

   -- Boundary conditions for configuration space.
   periodicDirs = {1,2}, -- Periodic directions.
   -- Integrated moment flag, compute quantities 1000 times in simulation.
   calcIntQuantEvery = 0.001,

   -- Electrons.
   elc = Plasma.Species {
      charge = -1.0, mass = 1.0,
      -- Velocity space grid.
      lower = vMinElc,
      upper = vMaxElc,
      cells = nvElc,
      decompCuts = {1,1},
      -- initial conditions
      init = function (t, xn)
         local x, y, vx, vy = xn[1], xn[2], xn[3], xn[4]
         local fv = maxwellian2v({vx, vy}, vDriftElc, vtElc)
	 return fv*(1)
      end,
      evolve = true, -- Evolve species?

      diagnosticMoments           = { "M0", "M1i" },
      diagnosticIntegratedMoments = { "intM0", "intM1i"},
      -- Collisions.
      coll = Plasma.LBOCollisions{
         collideWith = {'elc', 'ion'},
	 frequencies = {nuee, nuei},
      }
   },

   -- Ions.
   ion = Plasma.Species {
      charge = 1.0, mass = massRatio,
      -- Velocity space grid.
      lower = vMinIon,
      upper = vMaxIon,
      cells = nvIon,
      decompCuts = {1,1},
      -- Initial conditions.
      init = function (t, xn)
         local x, y, vx, vy = xn[1], xn[2], xn[3], xn[4]
         local fv = maxwellian2v({vx, vy}, vDriftIon, vtIon)
	 return fv       -- *(1.0 + perturbation*DrandKick({x,y},lx,nx,pOrder) ) RLW: I removed the noise
      end,
      evolve = true,    -- Evolve species?

      diagnosticMoments           = { "M0", "M1i" },
      diagnosticIntegratedMoments = { "intM0", "intM1i", "intM2Flow", "intM2Thermal", "intL2" },
      -- Collisions.
      coll = Plasma.LBOCollisions{
         collideWith = {'elc', 'ion'},
	 frequencies = {nuie, nuii},
      }
   },

   -- Field solver.
   field = Plasma.Field {
      epsilon0 = 1.0, mu0 = 1.0,
      init = function (t, xn)
         local x, y, vx, vy = xn[1], xn[2], xn[3], xn[4]
         local Ex = Ezero          --perturbation*randKickAntenna({x, y},lx,nx,pOrder) RLW: I removed the noise
         return Ex, 0.0, 0.0, 0.0, 0.0, 0.0
      end,
      evolve = true, -- Evolve field?
   },

   -- Current antenna.
   driveSpecies = Plasma.FuncSpecies {
      charge = 1.0, mass = 1.0,
      momentumDensity = function (t, xn)
         local x, y, vx, vy = xn[1], xn[2], xn[3], xn[4]
         local Jx = (vDriftElc[1] - vDriftIon[1]) + vTe*alpha*t  --note: the antenna current should have the opposite sign of the real current
	 local dJx =  FluctuatingCurrentDrive({x, y}, t, lx, nx, pOrder)[1]
	 local dJy =  FluctuatingCurrentDrive({x, y}, t, lx, nx, pOrder)[2]
         -- local dJx, dJy =  FluctuatingCurrentDrive({x, y}, t, lx, nx, pOrder)
         return Jx + noise*dJx, noise*dJy
      end,
      evolve = true, -- Evolve field?
   },
}
-- Run application.
plasmaApp:run()
